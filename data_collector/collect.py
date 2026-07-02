"""
Real-time Kalshi trade + orderbook data collector.

Connects to the Kalshi WebSocket for live trade events and polls the REST API
for orderbook snapshots of active markets.  Buffers data in memory and flushes
hourly Parquet files to S3 (with a local backup).

Usage:
    # Dry-run: print to stdout, no S3
    python -m data_collector.collect --dry-run

    # Write to S3
    python -m data_collector.collect --bucket kalshi-data-prod

    # Custom poll interval and flush interval
    python -m data_collector.collect --bucket kalshi-data-prod --poll-interval 30 --flush-interval 1800
"""

import argparse
import asyncio
import io
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import requests

from authentication_to_kalshi.auth import load_private_key, sign_pss, make_ws_headers
from config import API_KEY, KEY_PATH, WS_URL, RECONNECT_BASE, RECONNECT_MAX

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("collector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Recommended production API host
REST_BASE_URL = "https://external-api.kalshi.com"
# Bulk orderbook endpoint (up to 100 tickers per request)
ORDERBOOK_BULK_PATH = "/trade-api/v2/markets/orderbooks"
BATCH_SIZE = 100
REQUEST_SLEEP = 0.04      # ~25 req/s, under 30 read/s rate limit
LOCAL_BACKUP_DIR = Path("output/collector_backup")

# ---------------------------------------------------------------------------
# REST API helper (reuses auth pattern from fetch_btc_markets.py)
# ---------------------------------------------------------------------------
# Cache the private key so we load it once, not per-request
_private_key = None


def _get_private_key():
    """Load and cache private key for REST API signing."""
    global _private_key
    if _private_key is None:
        _private_key = load_private_key(KEY_PATH)
    return _private_key


def api_get(path: str, params: Optional[Dict] = None) -> Dict:
    """
    Authenticated GET request to Kalshi REST API.
    Signature = RSA-PSS(timestamp_ms + "GET" + path).
    """
    key = _get_private_key()
    timestamp_ms = str(int(time.time() * 1000))
    signature = sign_pss(key, timestamp_ms + "GET" + path)

    headers = {
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "Content-Type": "application/json",
    }
    resp = requests.get(
        REST_BASE_URL + path, headers=headers, params=params, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Orderbook parsing (reused from mm_fetch_orderbooks.py)
# ---------------------------------------------------------------------------
def _parse_levels(levels: list) -> tuple:
    """
    Parse one side of the orderbook (yes_dollars or no_dollars).
    Levels are [price_str, size_str] sorted ascending; best bid = last.
    Returns (best_price, best_size, total_depth, num_levels).
    """
    if not levels:
        return 0.0, 0.0, 0.0, 0
    best_price = float(levels[-1][0])
    best_size = float(levels[-1][1])
    total_depth = sum(float(lv[1]) for lv in levels)
    return best_price, best_size, total_depth, len(levels)


def _parse_orderbook(ticker: str, ob: dict, fetched_at: str) -> Optional[dict]:
    """
    Extract top-of-book, depth, and spread from an orderbook response.
    Returns None if both sides empty (settled market).
    """
    book = ob.get("orderbook_fp", ob)
    yes_levels = book.get("yes_dollars", book.get("yes", []))
    no_levels = book.get("no_dollars", book.get("no", []))

    if not yes_levels and not no_levels:
        return None

    yes_bid, yes_size, yes_depth, yes_n = _parse_levels(yes_levels)
    no_bid, no_size, no_depth, no_n = _parse_levels(no_levels)

    # Implied spread: YES ask (= 1 - NO bid) minus YES bid
    if yes_bid > 0 and no_bid > 0:
        implied_spread = (1.0 - no_bid) - yes_bid
    else:
        implied_spread = 0.0

    return {
        "ticker": ticker,
        "yes_best_bid": yes_bid,
        "yes_best_size": yes_size,
        "no_best_bid": no_bid,
        "no_best_size": no_size,
        "implied_spread": round(implied_spread, 4),
        "yes_depth_total": yes_depth,
        "no_depth_total": no_depth,
        "yes_levels": yes_n,
        "no_levels": no_n,
        "fetched_at": fetched_at,
    }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class DataCollector:
    """
    Buffers live trades and periodic orderbook snapshots, flushes to S3
    (and local disk) on a configurable interval.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: str = "us-east-2",
        poll_interval: int = 60,
        flush_interval: int = 300,  # 5 min default to avoid OOM on small instances
        dry_run: bool = False,
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.poll_interval = poll_interval    # seconds between orderbook polls
        self.flush_interval = flush_interval  # seconds between S3 flushes
        self.dry_run = dry_run

        # In-memory buffers
        self._trades: List[dict] = []
        self._orderbooks: List[dict] = []

        # Tickers seen in current flush window (only poll these)
        self._active_tickers: Set[str] = set()

        # S3 client (lazy init)
        self._s3 = None

        # Shutdown flag
        self._shutdown = False

    # ---- S3 ----
    def _get_s3(self):
        """Lazy-init boto3 S3 client."""
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    # ---- Trade callback (called from WS listener) ----
    def on_trade(self, msg: dict) -> None:
        """
        Handle a trade message from the WebSocket.
        Kalshi WS trade fields: market_ticker, yes_price_dollars, no_price_dollars,
        count_fp, taker_outcome_side, ts_ms.
        """
        # Kalshi WS trade messages have a "msg" wrapper with trade data
        trade_data = msg.get("msg", msg)
        ticker = trade_data.get("market_ticker", "")

        # Parse price and count (may be string or numeric)
        yes_price = float(trade_data.get("yes_price_dollars", 0))
        count = int(float(trade_data.get("count_fp", trade_data.get("count", 0))))
        taker_side = trade_data.get("taker_outcome_side", trade_data.get("taker_side", ""))
        ts = int(trade_data.get("ts_ms", trade_data.get("ts", time.time() * 1000)))

        row = {
            "ticker": ticker,
            "yes_price": yes_price,
            "count": count,
            "taker_side": taker_side,
            "ts": ts,
            "created_time": datetime.now(timezone.utc).isoformat(),
        }
        self._trades.append(row)
        self._active_tickers.add(ticker)

        if self.dry_run:
            log.info("TRADE  %s  price=%.2f  count=%d  side=%s",
                     ticker, yes_price, count, taker_side)

    # ---- Orderbook poller ----
    def poll_orderbooks(self) -> int:
        """
        Fetch orderbook snapshots for all active tickers via bulk REST API.
        Endpoint: /trade-api/v2/markets/orderbooks?tickers=A&tickers=B
        Returns number of orderbooks fetched.
        """
        tickers = sorted(self._active_tickers)
        if not tickers:
            return 0

        fetched = 0
        # Process in batches of up to 100 tickers
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i : i + BATCH_SIZE]
            fetched_at = datetime.now(timezone.utc).isoformat()

            try:
                # requests serializes list as tickers=A&tickers=B
                data = api_get(ORDERBOOK_BULK_PATH, {"tickers": batch})
            except Exception as e:
                log.warning("Orderbook fetch failed for batch %d: %s",
                            i // BATCH_SIZE, e)
                time.sleep(REQUEST_SLEEP)
                continue

            orderbooks = data.get("orderbooks", {})

            # Handle dict-keyed response {ticker: orderbook}
            if isinstance(orderbooks, dict):
                for tk, ob in orderbooks.items():
                    row = _parse_orderbook(tk, ob, fetched_at)
                    if row:
                        self._orderbooks.append(row)
                        fetched += 1
            # Handle list response [{ticker, ...}, ...]
            elif isinstance(orderbooks, list):
                for ob in orderbooks:
                    tk = ob.get("ticker", "")
                    row = _parse_orderbook(tk, ob, fetched_at)
                    if row:
                        self._orderbooks.append(row)
                        fetched += 1

            time.sleep(REQUEST_SLEEP)

        if self.dry_run and fetched > 0:
            log.info("ORDERBOOK  polled %d tickers, got %d snapshots",
                     len(tickers), fetched)

        return fetched

    # ---- Flush to S3 + local backup ----
    def flush(self) -> None:
        """
        Write buffered trades and orderbooks as Parquet to S3 and local disk.
        Clears buffers and active ticker set after successful write.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")

        trades_written = self._flush_buffer(
            self._trades,
            f"trades/{date_str}/trades_{hour_str}.parquet",
            "trades",
        )
        ob_written = self._flush_buffer(
            self._orderbooks,
            f"orderbooks/{date_str}/orderbooks_{hour_str}.parquet",
            "orderbooks",
        )

        log.info("FLUSH  trades=%d  orderbooks=%d  date=%s  hour=%s",
                 trades_written, ob_written, date_str, hour_str)

        # Clear buffers for next window
        self._trades.clear()
        self._orderbooks.clear()
        self._active_tickers.clear()

    def _flush_buffer(self, buf: List[dict], s3_key: str, label: str) -> int:
        """Write a buffer to S3 and local backup. Returns row count."""
        if not buf:
            return 0

        df = pl.DataFrame(buf)
        # Serialize to in-memory Parquet bytes
        buf_io = io.BytesIO()
        df.write_parquet(buf_io)
        parquet_bytes = buf_io.getvalue()

        # Local backup
        local_path = LOCAL_BACKUP_DIR / s3_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(parquet_bytes)

        # S3 upload
        if self.bucket and not self.dry_run:
            try:
                s3 = self._get_s3()
                s3.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=parquet_bytes,
                    ContentType="application/octet-stream",
                )
                log.info("S3  s3://%s/%s  (%d rows)", self.bucket, s3_key, len(df))
            except Exception as e:
                log.error("S3 upload failed for %s: %s  (local backup OK at %s)",
                          s3_key, e, local_path)
        else:
            log.info("LOCAL  %s  (%d rows)", local_path, len(df))

        return len(df)

    # ---- Graceful shutdown ----
    def request_shutdown(self) -> None:
        """Signal the collector to flush and exit."""
        log.info("Shutdown requested — flushing buffers...")
        self._shutdown = True


# ---------------------------------------------------------------------------
# WebSocket listener (adapted from websocket_client.py)
# ---------------------------------------------------------------------------
class _TradeWebSocket:
    """
    Minimal WebSocket client that subscribes to the global trade channel
    and forwards all trade events to the collector.
    """

    def __init__(self, collector: DataCollector) -> None:
        self._collector = collector
        self._key = load_private_key(KEY_PATH)
        self._ws = None

    async def run(self) -> None:
        """Connect with exponential-backoff reconnection loop."""
        delay = RECONNECT_BASE
        while not self._collector._shutdown:
            try:
                headers = make_ws_headers(API_KEY, self._key)
                import websockets
                self._ws = await websockets.connect(
                    WS_URL, additional_headers=headers,
                )
                log.info("WS connected to %s", WS_URL)

                # Subscribe to global trade channel (all markets)
                await self._ws.send(json.dumps({
                    "id": 1, "cmd": "subscribe",
                    "params": {"channels": ["trade"]},
                }))
                log.info("WS subscribed to global trade channel")

                delay = RECONNECT_BASE
                await self._listen()

            except Exception as e:
                if self._collector._shutdown:
                    break
                log.warning("WS error: %s — reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)

    async def _listen(self) -> None:
        """Read WS messages, dispatch trades to collector."""
        async for raw in self._ws:
            if self._collector._shutdown:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            # Trade event — forward to collector
            if msg_type == "trade":
                self._collector.on_trade(msg)


# ---------------------------------------------------------------------------
# Async scheduler tasks
# ---------------------------------------------------------------------------
async def _orderbook_poller(collector: DataCollector) -> None:
    """Poll orderbooks for active tickers on a fixed interval."""
    while not collector._shutdown:
        await asyncio.sleep(collector.poll_interval)
        if collector._shutdown:
            break
        # Run blocking REST calls in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        try:
            count = await loop.run_in_executor(None, collector.poll_orderbooks)
            log.debug("Polled %d orderbooks", count)
        except Exception:
            log.exception("Orderbook poll error")


async def _periodic_flusher(collector: DataCollector) -> None:
    """Flush buffers to S3 on a fixed interval."""
    while not collector._shutdown:
        await asyncio.sleep(collector.flush_interval)
        if collector._shutdown:
            break
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, collector.flush)
        except Exception:
            log.exception("Flush error — data NOT written to S3")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _run(args: argparse.Namespace) -> None:
    """Main async entry point: run WS listener, orderbook poller, and flusher."""
    collector = DataCollector(
        bucket=args.bucket,
        region=args.region,
        poll_interval=args.poll_interval,
        flush_interval=args.flush_interval,
        dry_run=args.dry_run,
    )

    # Wire up SIGTERM / SIGINT for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, collector.request_shutdown)

    log.info("Starting collector  dry_run=%s  bucket=%s  poll=%ds  flush=%ds",
             args.dry_run, args.bucket, args.poll_interval, args.flush_interval)

    # Launch concurrent tasks
    ws = _TradeWebSocket(collector)
    tasks = [
        asyncio.create_task(ws.run()),
        asyncio.create_task(_orderbook_poller(collector)),
        asyncio.create_task(_periodic_flusher(collector)),
    ]

    # Wait until shutdown is requested
    while not collector._shutdown:
        await asyncio.sleep(1)

    # Final flush
    collector.flush()

    # Cancel background tasks
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    log.info("Collector stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time Kalshi trade + orderbook collector",
    )
    parser.add_argument(
        "--bucket", type=str,
        default=None,
        help="S3 bucket name (omit for local-only or dry-run)",
    )
    parser.add_argument(
        "--region", type=str,
        default="us-east-2",
        help="AWS region (default: us-east-2)",
    )
    parser.add_argument(
        "--poll-interval", type=int,
        default=60,
        help="Seconds between orderbook polls (default: 60)",
    )
    parser.add_argument(
        "--flush-interval", type=int,
        default=300,
        help="Seconds between S3 flushes (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print data to stdout, skip S3 upload",
    )
    args = parser.parse_args()

    # Validate: need a bucket unless dry-run
    if not args.dry_run and not args.bucket:
        log.warning("No --bucket specified and not --dry-run; writing local backups only")
    if args.bucket:
        log.info("S3 target: s3://%s/  region=%s", args.bucket, args.region)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
