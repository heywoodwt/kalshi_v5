# AWS Production Deployment Plan — Kalshi Market Making Bot

**Date:** 2026-06-29
**Region:** us-east-2 (Ohio)
**Target:** Live trading with Kalshi WebSocket API
**Goal:** Deploy proven models for real-time market making

---

## Executive Summary

Deploy PPO-trained market-making agents on AWS us-east-2 with Kalshi WebSocket for real-time orderbook data and order execution. Start with paper trading (Phase 1), scale to live trading across 5 categories.

**Infrastructure:** AWS EC2, Docker, PostgreSQL, InfluxDB, Grafana
**Latency target:** <50ms to Kalshi API
**Uptime target:** 99.9% (4 hours downtime/year)
**Capital:** $500 at full scale

---

## 1. AWS Infrastructure Design

### Architecture Overview

```
┌──────────────────────────────────────────────────┐
│  AWS us-east-2 (Ohio)                            │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  EC2 Instance (c6i.xlarge)                 │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │  Docker Compose Stack                │  │  │
│  │  │                                      │  │  │
│  │  │  ┌────────────────┐                 │  │  │
│  │  │  │ Kalshi Client  │◄────────────────┼──┼──┼─► Kalshi WebSocket
│  │  │  │ - WebSocket    │                 │  │  │   (orderbook feed)
│  │  │  │ - REST API     │                 │  │  │
│  │  │  └────────┬───────┘                 │  │  │
│  │  │           │                         │  │  │
│  │  │  ┌────────▼────────┐                │  │  │
│  │  │  │ Data Pipeline   │                │  │  │
│  │  │  │ - Normalize     │                │  │  │
│  │  │  │ - 1-min windows │                │  │  │
│  │  │  └────────┬────────┘                │  │  │
│  │  │           │                         │  │  │
│  │  │  ┌────────▼────────┐                │  │  │
│  │  │  │ MM Agent        │                │  │  │
│  │  │  │ - Load models   │                │  │  │
│  │  │  │ - Inference     │                │  │  │
│  │  │  │ - Quote gen     │                │  │  │
│  │  │  └────────┬────────┘                │  │  │
│  │  │           │                         │  │  │
│  │  │  ┌────────▼────────┐                │  │  │
│  │  │  │ Order Manager   │                │  │  │
│  │  │  │ - Place/Cancel  │                │  │  │
│  │  │  │ - Track fills   │                │  │  │
│  │  │  │ - Inventory     │                │  │  │
│  │  │  └────────┬────────┘                │  │  │
│  │  │           │                         │  │  │
│  │  │  ┌────────▼────────┐                │  │  │
│  │  │  │ Risk Manager    │                │  │  │
│  │  │  │ - Loss limits   │                │  │  │
│  │  │  │ - Position      │                │  │  │
│  │  │  │ - Emergency     │                │  │  │
│  │  │  └─────────────────┘                │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  RDS PostgreSQL (db.t3.micro)              │  │
│  │  - Trade logs                              │  │
│  │  - Order history                           │  │
│  │  - PnL records                             │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  EC2 Monitoring Instance (t3.small)        │  │
│  │  - InfluxDB (time-series metrics)          │  │
│  │  - Grafana (dashboard)                     │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌────────────────────────────────────────────┐  │
│  │  S3 Bucket                                 │  │
│  │  - Model checkpoints                       │  │
│  │  - Backup logs                             │  │
│  │  - Historical data                         │  │
│  └────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### EC2 Instance Specifications

#### Trading Instance (Primary)

**Instance type:** `c6i.xlarge` (compute-optimized)
- **vCPUs:** 4
- **RAM:** 8 GB
- **Network:** Up to 12.5 Gbps
- **Cost:** ~$0.17/hour = $122/month (reserved: $73/month)

**Why us-east-2:**
- Low latency to Kalshi servers (likely east coast)
- Lower cost than us-east-1
- Good availability zone options

**AMI:** Ubuntu 22.04 LTS

**Storage:**
- Root: 30 GB gp3 SSD
- Data: 100 GB gp3 SSD (logs, models)

#### Monitoring Instance (Secondary)

**Instance type:** `t3.small`
- **vCPUs:** 2
- **RAM:** 2 GB
- **Cost:** ~$0.02/hour = $15/month

**Purpose:**
- InfluxDB (time-series metrics)
- Grafana (dashboard)
- CloudWatch agent

### Database

**RDS PostgreSQL:** `db.t3.micro`
- **vCPUs:** 2
- **RAM:** 1 GB
- **Storage:** 20 GB gp3
- **Cost:** ~$15/month

**Schema:**
```sql
-- Trade execution logs
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL,  -- 'buy' or 'sell'
    price DECIMAL(10, 4) NOT NULL,
    size INT NOT NULL,
    fee DECIMAL(10, 4) NOT NULL,
    pnl DECIMAL(10, 4),
    category VARCHAR(50),
    INDEX idx_timestamp (timestamp),
    INDEX idx_ticker (ticker)
);

-- Order history
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    ticker VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DECIMAL(10, 4) NOT NULL,
    size INT NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'placed', 'filled', 'cancelled'
    fill_timestamp TIMESTAMP,
    INDEX idx_order_id (order_id),
    INDEX idx_status (status)
);

-- Position tracking
CREATE TABLE positions (
    ticker VARCHAR(50) PRIMARY KEY,
    quantity INT NOT NULL,
    avg_price DECIMAL(10, 4),
    unrealized_pnl DECIMAL(10, 4),
    realized_pnl DECIMAL(10, 4),
    last_updated TIMESTAMP NOT NULL
);

-- Daily PnL summary
CREATE TABLE daily_pnl (
    date DATE PRIMARY KEY,
    category VARCHAR(50),
    gross_pnl DECIMAL(10, 4),
    fees DECIMAL(10, 4),
    net_pnl DECIMAL(10, 4),
    num_trades INT,
    INDEX idx_date (date)
);
```

### Storage (S3)

**Bucket:** `kalshi-mm-production`

**Structure:**
```
s3://kalshi-mm-production/
├── models/
│   ├── KXACBGAME.zip
│   ├── KXATP.zip
│   ├── KXATPCHALLENGERMATCH.zip
│   ├── KXAFCCLGAME.zip
│   ├── KXAPFDDH.zip
│   └── KXBTCD.zip
├── logs/
│   ├── 2026-06-29/
│   │   ├── trading.log
│   │   ├── errors.log
│   │   └── orders.log
│   └── ...
└── backups/
    ├── postgres_2026-06-29.sql.gz
    └── ...
```

**Lifecycle policy:**
- Logs: Move to Glacier after 30 days
- Backups: Keep for 90 days

**Cost:** ~$5/month (100 GB storage + transfers)

---

## 2. Kalshi WebSocket Integration

### WebSocket API Overview

**Kalshi WebSocket endpoints:**
- Production: `wss://trading-api.kalshi.com/trade-api/ws/v2`
- Sandbox: `wss://demo-api.kalshi.co/trade-api/ws/v2`

**Subscriptions needed:**
1. **Orderbook** — Real-time bid/ask depth
2. **Trades** — Market trades
3. **Orders** — Our order status updates
4. **Fills** — Our fills

### WebSocket Client Implementation

**Technology:** Python `websockets` library + `asyncio`

**File:** `kalshi_client/websocket_client.py`

```python
import asyncio
import json
import websockets
from typing import Callable, Dict, List
import logging

logger = logging.getLogger(__name__)

class KalshiWebSocketClient:
    """Kalshi WebSocket client for real-time market data."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: str = "production",
    ):
        self.api_key = api_key
        self.api_secret = api_secret

        if environment == "production":
            self.ws_url = "wss://trading-api.kalshi.com/trade-api/ws/v2"
        else:
            self.ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

        self.ws = None
        self.subscriptions: Dict[str, Callable] = {}
        self.running = False

    async def connect(self):
        """Establish WebSocket connection with authentication."""
        logger.info(f"Connecting to {self.ws_url}")

        self.ws = await websockets.connect(
            self.ws_url,
            extra_headers={
                "Authorization": f"Bearer {self._get_token()}",
            },
            ping_interval=20,  # Keep connection alive
            ping_timeout=10,
        )

        logger.info("WebSocket connected")
        self.running = True

    async def subscribe_orderbook(self, ticker: str, callback: Callable):
        """Subscribe to orderbook updates for a ticker."""
        msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": ticker,
            },
        }

        await self.ws.send(json.dumps(msg))
        self.subscriptions[f"orderbook:{ticker}"] = callback
        logger.info(f"Subscribed to orderbook for {ticker}")

    async def subscribe_trades(self, ticker: str, callback: Callable):
        """Subscribe to trade updates for a ticker."""
        msg = {
            "id": 2,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_ticker": ticker,
            },
        }

        await self.ws.send(json.dumps(msg))
        self.subscriptions[f"trades:{ticker}"] = callback
        logger.info(f"Subscribed to trades for {ticker}")

    async def subscribe_fills(self, callback: Callable):
        """Subscribe to our fill updates."""
        msg = {
            "id": 3,
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"],
            },
        }

        await self.ws.send(json.dumps(msg))
        self.subscriptions["fills"] = callback
        logger.info("Subscribed to fills")

    async def listen(self):
        """Listen for WebSocket messages and route to callbacks."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.running = False
            await self.reconnect()

    async def _handle_message(self, data: dict):
        """Route WebSocket message to appropriate callback."""
        msg_type = data.get("type")

        if msg_type == "orderbook_delta":
            ticker = data.get("market_ticker")
            callback = self.subscriptions.get(f"orderbook:{ticker}")
            if callback:
                await callback(data)

        elif msg_type == "ticker":
            ticker = data.get("market_ticker")
            callback = self.subscriptions.get(f"trades:{ticker}")
            if callback:
                await callback(data)

        elif msg_type == "fill":
            callback = self.subscriptions.get("fills")
            if callback:
                await callback(data)

    async def reconnect(self, max_retries: int = 5):
        """Reconnect with exponential backoff."""
        for attempt in range(max_retries):
            wait_time = 2 ** attempt
            logger.info(f"Reconnecting in {wait_time}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait_time)

            try:
                await self.connect()
                # Re-subscribe to all channels
                for sub_key, callback in self.subscriptions.items():
                    if sub_key.startswith("orderbook:"):
                        ticker = sub_key.split(":")[1]
                        await self.subscribe_orderbook(ticker, callback)
                    elif sub_key.startswith("trades:"):
                        ticker = sub_key.split(":")[1]
                        await self.subscribe_trades(ticker, callback)
                    elif sub_key == "fills":
                        await self.subscribe_fills(callback)

                logger.info("Reconnected and re-subscribed successfully")
                return
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt+1} failed: {e}")

        logger.critical("Failed to reconnect after max retries")
        raise Exception("WebSocket reconnection failed")

    def _get_token(self) -> str:
        """Get authentication token from Kalshi REST API."""
        # Implement OAuth token fetch
        # This would call Kalshi's /trade-api/v2/login endpoint
        pass

    async def close(self):
        """Close WebSocket connection gracefully."""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("WebSocket closed")
```

### Orderbook Processing

**File:** `data_pipeline/orderbook_processor.py`

```python
import polars as pl
from collections import deque
from typing import Dict
import time

class OrderbookProcessor:
    """Process WebSocket orderbook updates into 1-minute windows."""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self.orderbooks: Dict[str, dict] = {}  # ticker -> current orderbook
        self.windows: Dict[str, deque] = {}  # ticker -> deque of 1-min windows

    async def handle_orderbook_update(self, data: dict):
        """Process orderbook delta update from WebSocket."""
        ticker = data["market_ticker"]
        timestamp = data["timestamp"]

        # Initialize if new ticker
        if ticker not in self.orderbooks:
            self.orderbooks[ticker] = {
                "bids": {},
                "asks": {},
                "timestamp": timestamp,
            }
            self.windows[ticker] = deque(maxlen=100)  # Keep 100 windows

        # Apply delta update
        for bid in data.get("bids", []):
            price = bid["price"]
            size = bid["size"]
            if size == 0:
                self.orderbooks[ticker]["bids"].pop(price, None)
            else:
                self.orderbooks[ticker]["bids"][price] = size

        for ask in data.get("asks", []):
            price = ask["price"]
            size = ask["size"]
            if size == 0:
                self.orderbooks[ticker]["asks"].pop(price, None)
            else:
                self.orderbooks[ticker]["asks"][price] = size

        self.orderbooks[ticker]["timestamp"] = timestamp

        # Check if we need to create a new window
        if self._should_create_window(ticker):
            self._create_window(ticker)

    def _should_create_window(self, ticker: str) -> bool:
        """Check if it's time to create a new 1-minute window."""
        current_time = time.time()

        if not self.windows[ticker]:
            return True

        last_window = self.windows[ticker][-1]
        time_since_last = current_time - last_window["timestamp"]

        return time_since_last >= self.window_size

    def _create_window(self, ticker: str):
        """Create a new 1-minute window summary."""
        ob = self.orderbooks[ticker]

        if not ob["bids"] or not ob["asks"]:
            return  # Skip if orderbook is empty

        best_bid = max(ob["bids"].keys())
        best_ask = min(ob["asks"].keys())

        bid_depth = sum(ob["bids"].values())
        ask_depth = sum(ob["asks"].values())

        window = {
            "timestamp": ob["timestamp"],
            "ticker": ticker,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid,
            "mid_price": (best_bid + best_ask) / 2,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": (bid_depth - ask_depth) / (bid_depth + ask_depth),
        }

        self.windows[ticker].append(window)

    def get_latest_windows(self, ticker: str, n: int = 100) -> pl.DataFrame:
        """Get last N windows for a ticker as DataFrame."""
        if ticker not in self.windows:
            return pl.DataFrame()

        windows = list(self.windows[ticker])[-n:]
        return pl.DataFrame(windows)
```

---

## 3. Model Serving Infrastructure

### Docker Compose Stack

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Trading bot
  trading-bot:
    build: .
    container_name: kalshi-mm-bot
    environment:
      - KALSHI_API_KEY=${KALSHI_API_KEY}
      - KALSHI_API_SECRET=${KALSHI_API_SECRET}
      - KALSHI_ENV=production
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=kalshi_mm
      - DB_USER=postgres
      - DB_PASSWORD=${DB_PASSWORD}
      - PHASE=1  # 1=paper, 2=live-single, 3=live-multi, 4=live-scaled
      - CATEGORIES=KXACBGAME  # Comma-separated for Phase 3+
      - CONTRACT_SIZE=1
      - MAX_INVENTORY=20
      - DAILY_LOSS_LIMIT=50
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - postgres
    networks:
      - mm-network

  # PostgreSQL
  postgres:
    image: postgres:15
    container_name: kalshi-mm-db
    environment:
      - POSTGRES_DB=kalshi_mm
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - mm-network

  # InfluxDB (metrics)
  influxdb:
    image: influxdb:2.7
    container_name: kalshi-mm-influx
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=${INFLUX_PASSWORD}
      - DOCKER_INFLUXDB_INIT_ORG=kalshi-mm
      - DOCKER_INFLUXDB_INIT_BUCKET=metrics
    volumes:
      - influx-data:/var/lib/influxdb2
    ports:
      - "8086:8086"
    networks:
      - mm-network

  # Grafana (dashboard)
  grafana:
    image: grafana/grafana:latest
    container_name: kalshi-mm-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    depends_on:
      - influxdb
    networks:
      - mm-network

volumes:
  postgres-data:
  influx-data:
  grafana-data:

networks:
  mm-network:
    driver: bridge
```

### Dockerfile

**File:** `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY kalshi_client/ ./kalshi_client/
COPY data_pipeline/ ./data_pipeline/
COPY rl_bot/ ./rl_bot/
COPY main.py .

# Download models from S3 (will be done at runtime)
# Models are mounted as volume from host

CMD ["python", "main.py"]
```

### Main Trading Loop

**File:** `main.py`

```python
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import polars as pl
from stable_baselines3 import PPO

from kalshi_client.websocket_client import KalshiWebSocketClient
from kalshi_client.rest_client import KalshiRestClient
from data_pipeline.orderbook_processor import OrderbookProcessor
from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv
from order_manager import OrderManager
from risk_manager import RiskManager
from metrics import MetricsCollector

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self):
        self.phase = int(os.getenv("PHASE", "1"))
        self.categories = os.getenv("CATEGORIES", "KXACBGAME").split(",")
        self.contract_size = int(os.getenv("CONTRACT_SIZE", "1"))

        # Clients
        self.ws_client = KalshiWebSocketClient(
            api_key=os.getenv("KALSHI_API_KEY"),
            api_secret=os.getenv("KALSHI_API_SECRET"),
            environment=os.getenv("KALSHI_ENV", "production"),
        )
        self.rest_client = KalshiRestClient(
            api_key=os.getenv("KALSHI_API_KEY"),
            api_secret=os.getenv("KALSHI_API_SECRET"),
        )

        # Components
        self.ob_processor = OrderbookProcessor()
        self.order_manager = OrderManager(self.rest_client, paper_trading=(self.phase == 1))
        self.risk_manager = RiskManager(
            daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "50")),
        )
        self.metrics = MetricsCollector()

        # Models
        self.models = {}
        self._load_models()

        self.running = False

    def _load_models(self):
        """Load PPO models from checkpoints."""
        models_dir = Path("/app/models")

        for category in self.categories:
            model_path = models_dir / f"{category}.zip"
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                continue

            logger.info(f"Loading model for {category}")
            # Note: env=None for loading, we'll create env when needed
            self.models[category] = PPO.load(model_path)
            logger.info(f"✓ Loaded {category} model")

    async def start(self):
        """Start the trading bot."""
        logger.info(f"Starting trading bot (Phase {self.phase})")

        # Connect WebSocket
        await self.ws_client.connect()

        # Subscribe to tickers for each category
        tickers = self._get_active_tickers()
        logger.info(f"Subscribing to {len(tickers)} tickers")

        for ticker in tickers:
            await self.ws_client.subscribe_orderbook(
                ticker,
                self.ob_processor.handle_orderbook_update,
            )
            await self.ws_client.subscribe_trades(
                ticker,
                self._handle_trade_update,
            )

        # Subscribe to fills
        if self.phase >= 2:  # Live trading only
            await self.ws_client.subscribe_fills(
                self._handle_fill,
            )

        self.running = True
        logger.info("Trading bot started")

        # Start main loop
        await asyncio.gather(
            self.ws_client.listen(),
            self._trading_loop(),
            self._monitoring_loop(),
        )

    async def _trading_loop(self):
        """Main trading loop - generate quotes every N seconds."""
        while self.running:
            try:
                # Check risk limits
                if not self.risk_manager.can_trade():
                    logger.warning("Risk limits exceeded, pausing trading")
                    await asyncio.sleep(60)
                    continue

                # Generate quotes for each ticker
                tickers = self._get_active_tickers()

                for ticker in tickers:
                    category = self._get_category_for_ticker(ticker)
                    model = self.models.get(category)

                    if not model:
                        continue

                    # Get latest market state
                    windows = self.ob_processor.get_latest_windows(ticker, n=100)
                    if len(windows) == 0:
                        continue

                    # Get current observation (convert windows to MMEnv observation)
                    obs = self._create_observation(ticker, windows)

                    # Get model prediction
                    action, _ = model.predict(obs, deterministic=True)

                    # Convert action to quote (bid/ask)
                    quote = self._action_to_quote(action, windows[-1])

                    # Place/update orders
                    await self.order_manager.update_quotes(ticker, quote, self.contract_size)

                # Wait before next iteration
                await asyncio.sleep(5)  # Quote every 5 seconds

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _monitoring_loop(self):
        """Send metrics to InfluxDB every 10 seconds."""
        while self.running:
            try:
                # Collect metrics
                positions = await self.order_manager.get_positions()
                pnl = await self.risk_manager.get_current_pnl()

                # Send to InfluxDB
                await self.metrics.record_pnl(pnl)
                await self.metrics.record_positions(positions)

                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _handle_trade_update(self, data: dict):
        """Handle trade update from WebSocket."""
        # Log trade, update metrics
        pass

    async def _handle_fill(self, data: dict):
        """Handle fill notification from WebSocket."""
        logger.info(f"Fill: {data}")
        await self.order_manager.process_fill(data)
        await self.risk_manager.update_pnl(data)
        await self.metrics.record_fill(data)

    def _get_active_tickers(self):
        """Get list of active tickers to trade."""
        # Query Kalshi API for active tickers in our categories
        # For now, return a placeholder
        return []

    def _get_category_for_ticker(self, ticker: str) -> str:
        """Determine category from ticker."""
        # Ticker format: CATEGORY-YYYYMMDD-...
        return ticker.split("-")[0]

    def _create_observation(self, ticker: str, windows: pl.DataFrame):
        """Convert orderbook windows to MMEnv observation."""
        # This would mirror the observation space from training
        # 16-dim: [position, time_to_expiry, spread, imbalance, ...]
        pass

    def _action_to_quote(self, action, latest_window) -> dict:
        """Convert PPO action to bid/ask quote."""
        # Action = [half_spread, skew]
        # Quote = {bid: price, ask: price, size: N}
        pass

    async def stop(self):
        """Gracefully shutdown the bot."""
        logger.info("Shutting down trading bot...")
        self.running = False

        # Cancel all orders
        await self.order_manager.cancel_all_orders()

        # Close WebSocket
        await self.ws_client.close()

        logger.info("Trading bot stopped")


async def main():
    bot = TradingBot()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(bot.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        await bot.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Deployment Steps

### Step 1: AWS Account Setup

```bash
# 1. Create AWS account (if needed)
# 2. Set up IAM user with programmatic access
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-2)

# 3. Create S3 bucket
aws s3 mb s3://kalshi-mm-production --region us-east-2

# 4. Upload model checkpoints
aws s3 cp rl_bot/mm_checkpoints/ s3://kalshi-mm-production/models/ --recursive
```

### Step 2: Launch EC2 Instances

**Trading instance:**

```bash
# Launch c6i.xlarge in us-east-2
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 (verify latest AMI)
    --instance-type c6i.xlarge \
    --key-name kalshi-mm-key \
    --security-group-ids sg-xxxxx \
    --subnet-id subnet-xxxxx \
    --region us-east-2 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kalshi-mm-trading}]'
```

**Monitoring instance:**

```bash
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.small \
    --key-name kalshi-mm-key \
    --security-group-ids sg-xxxxx \
    --subnet-id subnet-xxxxx \
    --region us-east-2 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kalshi-mm-monitoring}]'
```

### Step 3: Setup RDS PostgreSQL

```bash
aws rds create-db-instance \
    --db-instance-identifier kalshi-mm-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.3 \
    --master-username postgres \
    --master-user-password "${DB_PASSWORD}" \
    --allocated-storage 20 \
    --storage-type gp3 \
    --vpc-security-group-ids sg-xxxxx \
    --db-subnet-group-name kalshi-mm-subnet-group \
    --region us-east-2
```

### Step 4: Configure Security Groups

```bash
# Trading instance security group
aws ec2 create-security-group \
    --group-name kalshi-mm-trading-sg \
    --description "Security group for Kalshi MM trading instance" \
    --vpc-id vpc-xxxxx

# Allow SSH (for deployment only)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0  # Restrict to your IP in production

# Allow outbound HTTPS (to Kalshi API)
aws ec2 authorize-security-group-egress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

# Allow PostgreSQL access from trading instance
aws ec2 authorize-security-group-ingress \
    --group-id sg-db-xxxxx \
    --protocol tcp \
    --port 5432 \
    --source-group sg-xxxxx
```

### Step 5: Install Software on EC2

**SSH to trading instance:**

```bash
ssh -i kalshi-mm-key.pem ubuntu@<trading-instance-ip>
```

**Install Docker:**

```bash
# Update packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt-get install -y docker-compose

# Logout and login to apply group changes
exit
ssh -i kalshi-mm-key.pem ubuntu@<trading-instance-ip>
```

### Step 6: Deploy Application

```bash
# Clone repository
git clone https://github.com/your-org/kalshi-mm.git
cd kalshi-mm

# Download models from S3
aws s3 sync s3://kalshi-mm-production/models/ ./models/

# Create .env file
cat > .env << EOF
KALSHI_API_KEY=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here
KALSHI_ENV=production
DB_PASSWORD=your_db_password
INFLUX_PASSWORD=your_influx_password
GRAFANA_PASSWORD=your_grafana_password
PHASE=1
CATEGORIES=KXACBGAME
CONTRACT_SIZE=0
LOG_LEVEL=INFO
EOF

# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f trading-bot
```

### Step 7: Verify Deployment

```bash
# Check containers running
docker-compose ps

# Check trading bot logs
docker-compose logs trading-bot | tail -100

# Check PostgreSQL connection
docker-compose exec postgres psql -U postgres -d kalshi_mm -c "SELECT COUNT(*) FROM orders;"

# Access Grafana dashboard
# http://<monitoring-instance-ip>:3000
# Login: admin / <GRAFANA_PASSWORD>
```

---

## 5. Monitoring & Alerting

### Grafana Dashboard

**Panels to include:**

1. **Current PnL** (gauge)
   - Today's PnL
   - Week's PnL
   - All-time PnL

2. **Position Overview** (table)
   - Ticker, Quantity, Avg Price, Unrealized PnL

3. **Fill Rate** (time series)
   - Quote fills per hour
   - Comparison to backtest expectation

4. **Order Flow** (time series)
   - Orders placed/cancelled/filled over time

5. **Risk Metrics** (gauges)
   - Current inventory utilization (% of max)
   - Daily loss remaining before limit
   - Largest position size

6. **Latency** (histogram)
   - Quote submission latency
   - Fill notification latency

### CloudWatch Alarms

```bash
# Create SNS topic for alerts
aws sns create-topic --name kalshi-mm-alerts --region us-east-2

# Subscribe email to topic
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-2:xxxxx:kalshi-mm-alerts \
    --protocol email \
    --notification-endpoint your-email@example.com

# Create alarm: Daily loss limit exceeded
aws cloudwatch put-metric-alarm \
    --alarm-name kalshi-mm-daily-loss-limit \
    --alarm-description "Daily PnL below -50" \
    --metric-name DailyPnL \
    --namespace KalshiMM \
    --statistic Average \
    --period 300 \
    --evaluation-periods 1 \
    --threshold -50 \
    --comparison-operator LessThanThreshold \
    --alarm-actions arn:aws:sns:us-east-2:xxxxx:kalshi-mm-alerts

# Create alarm: EC2 instance down
aws cloudwatch put-metric-alarm \
    --alarm-name kalshi-mm-instance-down \
    --alarm-description "Trading instance stopped" \
    --metric-name StatusCheckFailed \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 60 \
    --evaluation-periods 2 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --dimensions Name=InstanceId,Value=i-xxxxx \
    --alarm-actions arn:aws:sns:us-east-2:xxxxx:kalshi-mm-alerts
```

---

## 6. Cost Breakdown

### Monthly Costs (Full Production)

| Service | Specification | Monthly Cost |
|---------|---------------|-------------:|
| EC2 Trading (c6i.xlarge) | On-demand | $122 |
| EC2 Trading (c6i.xlarge) | Reserved 1-year | $73 |
| EC2 Monitoring (t3.small) | On-demand | $15 |
| RDS PostgreSQL (db.t3.micro) | 20 GB | $15 |
| S3 Storage | 100 GB + transfers | $5 |
| Data Transfer | 100 GB/month | $9 |
| CloudWatch | Logs + alarms | $10 |
| **Total (on-demand)** | | **$176/month** |
| **Total (reserved)** | | **$127/month** |

**Annual:** $1,524-$2,112

**Cost per $100k revenue:** 1.5-2.1% (infrastructure only)

### Cost Optimization

**Reserved instances (1-year):**
- c6i.xlarge: $73/month (40% savings)
- Total savings: ~$600/year

**Spot instances (risky for trading):**
- Not recommended (can be terminated)

**Right-sizing after Phase 1:**
- If Phase 1 paper trading shows low CPU usage, downgrade to c6i.large ($61/month)

---

## 7. Disaster Recovery

### Backup Strategy

**Automated backups:**

```bash
# Daily PostgreSQL backup to S3
0 2 * * * docker-compose exec postgres pg_dump -U postgres kalshi_mm | gzip | aws s3 cp - s3://kalshi-mm-production/backups/postgres_$(date +\%Y-\%m-\%d).sql.gz
```

**Retention:**
- Daily backups: 30 days
- Weekly backups: 90 days
- Monthly backups: 1 year

### Recovery Procedures

**If trading instance fails:**

1. Launch new EC2 instance from AMI snapshot
2. Mount EBS volume with models/logs
3. Deploy code via git pull
4. Restart Docker containers
5. Verify WebSocket connection
6. Resume trading

**Estimated recovery time:** <15 minutes

**If database corrupted:**

1. Launch new RDS instance
2. Restore from latest S3 backup
3. Update trading instance DB_HOST
4. Verify data integrity
5. Resume trading

**Estimated recovery time:** <30 minutes

### High Availability (Future)

**For 99.99% uptime:**

- Multi-AZ RDS deployment
- Auto Scaling Group with 2+ trading instances
- Application Load Balancer (for monitoring only)
- Cross-region S3 replication

**Additional cost:** +$100-150/month

---

## 8. Security Considerations

### API Key Management

**Use AWS Secrets Manager:**

```bash
# Store Kalshi API credentials
aws secretsmanager create-secret \
    --name kalshi-mm/api-credentials \
    --secret-string '{"api_key":"xxx","api_secret":"xxx"}' \
    --region us-east-2

# Update main.py to fetch from Secrets Manager
import boto3
secrets = boto3.client('secretsmanager', region_name='us-east-2')
response = secrets.get_secret_value(SecretId='kalshi-mm/api-credentials')
credentials = json.loads(response['SecretString'])
```

**Cost:** $0.40/month per secret

### Network Security

- VPC with private subnets for RDS
- Security groups: least privilege access
- SSH access: restricted to admin IPs only
- API access: HTTPS only
- VPN: Recommended for production access

### Logging & Audit

- All API calls logged to CloudWatch
- Trade executions logged to PostgreSQL
- Order history immutable (append-only)
- Daily audit reports

---

## 9. Phase Rollout on AWS

### Phase 1: Paper Trading (Weeks 1-4)

**Configuration:**
```bash
PHASE=1
CATEGORIES=KXACBGAME
CONTRACT_SIZE=0  # Paper trading, no real orders
```

**Goal:** Validate WebSocket integration, measure latency

**Success criteria:**
- Bot runs 24/7 for 4 weeks
- WebSocket uptime >99.5%
- Latency <50ms average
- Fill rate ≥15% of backtest

### Phase 2: Single-Category Live (Weeks 5-8)

**Configuration:**
```bash
PHASE=2
CATEGORIES=KXACBGAME
CONTRACT_SIZE=1
```

**Goal:** Validate profitability on lowest-risk category

**Success criteria:**
- Cumulative PnL >$0 after 20 days
- Live $/episode ≥20% of backtest

### Phase 3: Multi-Category (Weeks 9-16)

**Configuration:**
```bash
PHASE=3
CATEGORIES=KXACBGAME,KXATP,KXATPCHALLENGERMATCH,KXAFCCLGAME,KXAPFDDH
CONTRACT_SIZE=1
```

**Goal:** Scale to 5 categories

**Expected:** $180/day at 20% efficiency

### Phase 4: Size Scaling (Weeks 17-24)

**Configuration:**
```bash
PHASE=4
CATEGORIES=KXACBGAME,KXATP,KXATPCHALLENGERMATCH,KXAFCCLGAME,KXAPFDDH
CONTRACT_SIZE=3
```

**Goal:** Reach $100k/year

**Expected:** $541/day at 20% efficiency

---

## 10. Next Steps

### Week 1: Infrastructure Setup
- [ ] Create AWS account, configure IAM
- [ ] Launch EC2 instances (trading + monitoring)
- [ ] Set up RDS PostgreSQL
- [ ] Create S3 bucket, upload models
- [ ] Configure security groups

### Week 2: Application Development
- [ ] Implement WebSocket client
- [ ] Implement orderbook processor
- [ ] Implement order manager
- [ ] Implement risk manager
- [ ] Write main.py trading loop

### Week 3: Deployment & Testing
- [ ] Deploy Docker stack to EC2
- [ ] Configure Grafana dashboards
- [ ] Set up CloudWatch alarms
- [ ] Test WebSocket connection
- [ ] Verify model loading

### Week 4: Phase 1 Launch (Paper Trading)
- [ ] Start paper trading bot
- [ ] Monitor for 4 weeks
- [ ] Collect fill rate data
- [ ] Measure latency
- [ ] Validate backtest assumptions

---

## Appendix A: Required Python Packages

**File:** `requirements.txt`

```txt
# Core
python-dotenv==1.0.0
polars==0.20.5
numpy==1.26.0

# RL
stable-baselines3==2.2.1
gymnasium==0.29.1
torch==2.1.0

# API & WebSocket
websockets==12.0
aiohttp==3.9.0
requests==2.31.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Metrics
influxdb-client==1.38.0

# AWS
boto3==1.34.0

# Logging
python-json-logger==2.0.7
```

---

## Appendix B: Kalshi API Credentials

**Obtain from Kalshi:**

1. Sign up at https://kalshi.com
2. Navigate to Settings → API
3. Generate API key and secret
4. Store in AWS Secrets Manager

**Sandbox vs Production:**

- Sandbox: https://demo-api.kalshi.co (for Phase 1 testing)
- Production: https://trading-api.kalshi.com (for Phase 2+)

---

**Ready to deploy? Start with Week 1 infrastructure setup.**
