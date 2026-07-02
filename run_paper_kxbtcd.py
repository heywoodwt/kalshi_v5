"""Bounded paper-trading session for the KXBTCD 20-dim model.

Runs the real live_trader_v2 pipeline in PAPER mode (orders simulated, market
data real/authenticated) for a fixed number of seconds, then stops. Safe on a
real-money account because the paper client simulates all order placement.

Usage:
    TRADING_CONFIG=rl_bot.live_config_kxbtcd_paper python run_paper_kxbtcd.py [seconds]
"""
import asyncio
import os
import sys

# Force paper mode regardless of .env (which defaults PAPER_MODE=false / live).
os.environ["PAPER_MODE"] = "true"
os.environ.setdefault("TRADING_CONFIG", "rl_bot.live_config_kxbtcd_paper")

from rl_bot.live_trader_v2 import LiveTrader  # noqa: E402 (env must be set first)

DURATION_S = int(sys.argv[1]) if len(sys.argv) > 1 else 90


async def main():
    trader = LiveTrader(paper_mode=True)
    assert trader.paper_mode, "SAFETY: trader is not in paper mode — aborting"
    try:
        # Bound the whole lifecycle so a hung WS/REST call cannot run forever.
        await asyncio.wait_for(_init_and_run(trader), timeout=DURATION_S)
    except asyncio.TimeoutError:
        print(f"\n=== PAPER RUN: reached {DURATION_S}s limit, stopping ===")
        trader.running = False


async def _init_and_run(trader):
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
