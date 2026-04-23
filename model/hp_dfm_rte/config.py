"""Pipeline configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    # Kalshi auth
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""

    # WebSocket / REST
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    rest_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    market_prefix: str = "KXBTC"

    # Accumulator
    bucket_interval_s: int = 60
    rolling_window: int = 300  # 300 buckets; at 5-min intervals = 25 hours of history
    min_observations: int = 10
    min_std_threshold: float = 0.01
    min_unique_prices: int = 5  # Drop tickers with fewer than N distinct prices

    # Model - HP-DFM-RTE parameters
    hp_lambda: float = 6.25
    k_factors: int = 2       # 2 factors for capturing market dynamics
    factor_order: int = 3    # AR(3) for factor transitions
    refit_interval_s: int = 60

    # EWMA model (replaces batch DFM when use_ewma=True)
    use_ewma: bool = False   # Use HP-DFM-RTE by default
    ewma_alpha: float = 0.1  # Smoothing factor; half-life ~6.6 observations

    # Signal thresholds
    signal_threshold: float = 0.005  # Min signal strength (z-score)
    cooldown_s: int = 120
    min_edge_cents: float = 3.0      # Require 3c expected profit minimum

    # Market-making quote policy
    mm_enabled: bool = True
    mm_tft_weight: float = 0.7
    mm_cycle_tft_weight: float = 0.7
    mm_min_edge_prob: float = 0.02           # 2 cents in [0,1] probability space
    mm_edge_fee_multiple: float = 2.0        # E = max(fee_multiple * fee, min_edge_prob)
    mm_cycle_shift_k: float = 0.02           # Quote shift from cycle score
    mm_cycle_norm_scale: float = 2.0         # Normalizes cycle to roughly [-1, 1]
    mm_subpenny_step_prob: float = 0.001     # Queue step (0.1 cent)
    mm_extreme_edge_multiple: float = 2.0    # Cross spread only if edge > multiple * E
    mm_size_edge_step_prob: float = 0.01     # Size step for edge-proportional sizing
    mm_base_contracts: int = 1

    # Signal filters
    price_filter_min: float = 20.0   # Min yes_price (cents) to consider trading
    price_filter_max: float = 80.0   # Max yes_price (cents) to consider trading
    z_score_price_scaling: bool = True
    min_expected_move_multiple: float = 2.0
    expiry_blackout_minutes: int = 15

    # Momentum filter
    momentum_filter_enabled: bool = True
    momentum_lookback: int = 3

    # Volatility-adaptive threshold
    vol_adaptive_threshold_enabled: bool = True
    vol_short_period: int = 5
    vol_medium_period: int = 20

    # Position sizing
    max_contracts: int = 3
    risk_per_trade_pct: float = 3.0
    initial_capital: float = 10000.0

    # Cross-market monotonicity arbitrage
    monotonicity_arb_enabled: bool = True
    monotonicity_min_spread: float = 2.0

    # Exit logic
    exit_z_threshold: float = 0.5
    exit_signals_enabled: bool = True
    min_price_movement_cents: float = 1.0
    holding_period_minutes: float = 20.0
    profit_target_cents: float = 5.0
    stop_loss_cents: float = 10.0
    trailing_stop_cents: float = 8.0
    trailing_stop_pct: float = 0.70
    expiry_exit_minutes: float = 2.0

    # Orderbook fillability gate
    orderbook_enabled: bool = True
    orderbook_max_age_s: int = 10
    orderbook_min_size: int = 1
    orderbook_slippage_cents: float = 0.0

    # Fees
    assume_maker: bool = True

    # Expectancy guardrail (used by backtests / sims)
    expectancy_win_rate: float = 0.55
    expectancy_avg_win_cents: float = 3.0
    expectancy_avg_loss_cents: float = 6.0

    # Trading / Order execution
    trading_enabled: bool = False
    paper_trading: bool = True
    order_count: int = 1
    max_open_orders: int = 10
    order_ttl_s: int = 300
    db_path: str = "./output/trades.db"
    api_rate_limit: int = 30

    # Historical seeding
    max_seed_markets: int = 100

    # Logging
    log_level: str = "INFO"
    log_file: str = ""

    @classmethod
    def with_profile(cls, profile_name: str) -> PipelineConfig:
        """Create a config with named profile overrides applied."""
        cfg = cls()
        profiles: dict[str, dict[str, float | bool]] = {
            "econ_v1": {
                "profit_target_cents": 3.0,
                "stop_loss_cents": 6.0,
                "assume_maker": False,
            },
        }
        overrides = profiles.get(profile_name)
        if overrides is None:
            raise ValueError(f"Unknown profile: {profile_name}")
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Load config from environment variables with .env fallback."""
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                # Strip inline comments before processing value
                value = value.split('#')[0].strip()
                key, value = key.strip(), value.strip("'\"")
                if key and value:
                    os.environ.setdefault(key, value)

        return cls(
            kalshi_api_key_id=os.environ.get(
                "KALSHI_API_KEY_ID", os.environ.get("PROD_API_KEY", "")
            ),
            kalshi_private_key_path=os.environ.get(
                "KALSHI_PRIVATE_KEY_PATH", os.environ.get("PROD_KEY_PATH", "")
            ),
            ws_url=os.environ.get("WS_URL", cls.ws_url),
            rest_url=os.environ.get("REST_URL", cls.rest_url),
            market_prefix=os.environ.get("MARKET_PREFIX", cls.market_prefix),
            bucket_interval_s=int(os.environ.get("BUCKET_INTERVAL_S", cls.bucket_interval_s)),
            rolling_window=int(os.environ.get("ROLLING_WINDOW", cls.rolling_window)),
            min_observations=int(os.environ.get("MIN_OBSERVATIONS", cls.min_observations)),
            min_std_threshold=float(os.environ.get("MIN_STD_THRESHOLD", cls.min_std_threshold)),
            min_unique_prices=int(os.environ.get("MIN_UNIQUE_PRICES", cls.min_unique_prices)),
            hp_lambda=float(os.environ.get("HP_LAMBDA", cls.hp_lambda)),
            k_factors=int(os.environ.get("K_FACTORS", cls.k_factors)),
            factor_order=int(os.environ.get("FACTOR_ORDER", cls.factor_order)),
            refit_interval_s=int(os.environ.get("REFIT_INTERVAL_S", cls.refit_interval_s)),
            use_ewma=os.environ.get("USE_EWMA", "false").lower() in ("true", "1", "yes"),
            ewma_alpha=float(os.environ.get("EWMA_ALPHA", cls.ewma_alpha)),
            signal_threshold=float(os.environ.get("SIGNAL_THRESHOLD", cls.signal_threshold)),
            cooldown_s=int(os.environ.get("COOLDOWN_S", cls.cooldown_s)),
            min_edge_cents=float(os.environ.get("MIN_EDGE_CENTS", cls.min_edge_cents)),
            mm_enabled=os.environ.get("MM_ENABLED", "true").lower() in ("true", "1", "yes"),
            mm_tft_weight=float(os.environ.get("MM_TFT_WEIGHT", cls.mm_tft_weight)),
            mm_cycle_tft_weight=float(os.environ.get("MM_CYCLE_TFT_WEIGHT", cls.mm_cycle_tft_weight)),
            mm_min_edge_prob=float(os.environ.get("MM_MIN_EDGE_PROB", cls.mm_min_edge_prob)),
            mm_edge_fee_multiple=float(os.environ.get("MM_EDGE_FEE_MULTIPLE", cls.mm_edge_fee_multiple)),
            mm_cycle_shift_k=float(os.environ.get("MM_CYCLE_SHIFT_K", cls.mm_cycle_shift_k)),
            mm_cycle_norm_scale=float(os.environ.get("MM_CYCLE_NORM_SCALE", cls.mm_cycle_norm_scale)),
            mm_subpenny_step_prob=float(os.environ.get("MM_SUBPENNY_STEP_PROB", cls.mm_subpenny_step_prob)),
            mm_extreme_edge_multiple=float(os.environ.get("MM_EXTREME_EDGE_MULTIPLE", cls.mm_extreme_edge_multiple)),
            mm_size_edge_step_prob=float(os.environ.get("MM_SIZE_EDGE_STEP_PROB", cls.mm_size_edge_step_prob)),
            mm_base_contracts=int(os.environ.get("MM_BASE_CONTRACTS", cls.mm_base_contracts)),
            price_filter_min=float(os.environ.get("PRICE_FILTER_MIN", cls.price_filter_min)),
            price_filter_max=float(os.environ.get("PRICE_FILTER_MAX", cls.price_filter_max)),
            z_score_price_scaling=os.environ.get("Z_SCORE_PRICE_SCALING", "true").lower() in ("true", "1", "yes"),
            min_expected_move_multiple=float(os.environ.get("MIN_EXPECTED_MOVE_MULTIPLE", cls.min_expected_move_multiple)),
            expiry_blackout_minutes=int(os.environ.get("EXPIRY_BLACKOUT_MINUTES", cls.expiry_blackout_minutes)),
            momentum_filter_enabled=os.environ.get("MOMENTUM_FILTER_ENABLED", "true").lower() in ("true", "1", "yes"),
            momentum_lookback=int(os.environ.get("MOMENTUM_LOOKBACK", cls.momentum_lookback)),
            vol_adaptive_threshold_enabled=os.environ.get("VOL_ADAPTIVE_THRESHOLD_ENABLED", "true").lower() in ("true", "1", "yes"),
            vol_short_period=int(os.environ.get("VOL_SHORT_PERIOD", cls.vol_short_period)),
            vol_medium_period=int(os.environ.get("VOL_MEDIUM_PERIOD", cls.vol_medium_period)),
            max_contracts=int(os.environ.get("MAX_CONTRACTS", cls.max_contracts)),
            risk_per_trade_pct=float(os.environ.get("RISK_PER_TRADE_PCT", cls.risk_per_trade_pct)),
            initial_capital=float(os.environ.get("INITIAL_CAPITAL", cls.initial_capital)),
            monotonicity_arb_enabled=os.environ.get("MONOTONICITY_ARB_ENABLED", "true").lower() in ("true", "1", "yes"),
            monotonicity_min_spread=float(os.environ.get("MONOTONICITY_MIN_SPREAD", cls.monotonicity_min_spread)),
            exit_z_threshold=float(os.environ.get("EXIT_Z_THRESHOLD", cls.exit_z_threshold)),
            exit_signals_enabled=os.environ.get("EXIT_SIGNALS_ENABLED", "true").lower() in ("true", "1", "yes"),
            min_price_movement_cents=float(os.environ.get("MIN_PRICE_MOVEMENT_CENTS", cls.min_price_movement_cents)),
            holding_period_minutes=float(os.environ.get("HOLDING_PERIOD_MINUTES", cls.holding_period_minutes)),
            profit_target_cents=float(os.environ.get("PROFIT_TARGET_CENTS", cls.profit_target_cents)),
            stop_loss_cents=float(os.environ.get("STOP_LOSS_CENTS", cls.stop_loss_cents)),
            trailing_stop_cents=float(os.environ.get("TRAILING_STOP_CENTS", cls.trailing_stop_cents)),
            trailing_stop_pct=float(os.environ.get("TRAILING_STOP_PCT", cls.trailing_stop_pct)),
            expiry_exit_minutes=float(os.environ.get("EXPIRY_EXIT_MINUTES", cls.expiry_exit_minutes)),
            orderbook_enabled=os.environ.get("ORDERBOOK_ENABLED", "true").lower() in ("true", "1", "yes"),
            orderbook_max_age_s=int(os.environ.get("ORDERBOOK_MAX_AGE_S", cls.orderbook_max_age_s)),
            orderbook_min_size=int(os.environ.get("ORDERBOOK_MIN_SIZE", cls.orderbook_min_size)),
            orderbook_slippage_cents=float(os.environ.get("ORDERBOOK_SLIPPAGE_CENTS", cls.orderbook_slippage_cents)),
            assume_maker=os.environ.get("ASSUME_MAKER", "true").lower() in ("true", "1", "yes"),
            expectancy_win_rate=float(os.environ.get("EXPECTANCY_WIN_RATE", cls.expectancy_win_rate)),
            expectancy_avg_win_cents=float(os.environ.get("EXPECTANCY_AVG_WIN_CENTS", cls.expectancy_avg_win_cents)),
            expectancy_avg_loss_cents=float(os.environ.get("EXPECTANCY_AVG_LOSS_CENTS", cls.expectancy_avg_loss_cents)),
            trading_enabled=os.environ.get("TRADING_ENABLED", "false").lower() in ("true", "1", "yes"),
            paper_trading=os.environ.get("PAPER_TRADING", "true").lower() in ("true", "1", "yes"),
            order_count=int(os.environ.get("ORDER_COUNT", cls.order_count)),
            max_open_orders=int(os.environ.get("MAX_OPEN_ORDERS", cls.max_open_orders)),
            order_ttl_s=int(os.environ.get("ORDER_TTL_S", cls.order_ttl_s)),
            db_path=os.environ.get("DB_PATH", cls.db_path),
            api_rate_limit=int(os.environ.get("API_RATE_LIMIT", cls.api_rate_limit)),
            max_seed_markets=int(os.environ.get("MAX_SEED_MARKETS", cls.max_seed_markets)),
            log_level=os.environ.get("LOG_LEVEL", cls.log_level),
            log_file=os.environ.get("LOG_FILE", ""),
        )
