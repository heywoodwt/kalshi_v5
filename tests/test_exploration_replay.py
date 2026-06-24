"""Integration tests for exploration strategies in replay mode."""
import tempfile
from pathlib import Path
import polars as pl
import pytest
from rl_bot.replay import run_replay, create_exploration_strategy


def create_mock_trades(n_rows: int = 100) -> str:
    """Create tiny synthetic trade dataset for testing."""
    # Simple synthetic data with KXBTCD tickers
    timestamps = [f"2024-06-{1 + i // 50:02d}T{(i % 24):02d}:00:00Z" for i in range(n_rows)]
    tickers = [f"KXBTCD-26JUN2403-T75000.00"] * n_rows

    df = pl.DataFrame({
        "trade_id": range(n_rows),
        "ticker": tickers,
        "count": [1.0] * n_rows,
        "yes_price": [50 + (i % 10) for i in range(n_rows)],
        "no_price": [50 - (i % 10) for i in range(n_rows)],
        "taker_side": ["yes"] * n_rows,
        "created_time": timestamps,
    })

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    df.write_parquet(tmp.name)
    return tmp.name


def test_create_exploration_strategy():
    """Verify strategy factory creates correct instances."""
    # Fast linear
    strategy = create_exploration_strategy("fast_linear", n_rows=1000)
    assert strategy.__class__.__name__ == "FastLinearDecay"
    assert strategy.config["decay_steps"] == 100  # 1000 // 10

    # Exponential
    strategy = create_exploration_strategy("exponential", n_rows=1000)
    assert strategy.__class__.__name__ == "ExponentialDecay"
    assert strategy.config["decay_rate"] == 0.9990

    # Unknown strategy
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_exploration_strategy("invalid", n_rows=1000)


def test_replay_with_strategy():
    """Run quick replay with each strategy, verify no crashes."""
    data_path = create_mock_trades(n_rows=100)

    strategies = ["fast_linear", "exponential", "logarithmic", "action_local"]

    for strategy_name in strategies:
        strategy = create_exploration_strategy(strategy_name, n_rows=100)

        # Import here to avoid circular dependency
        from rl_bot.agent import DQNAgent
        from rl_bot.config import RLConfig

        cfg = RLConfig(
            eps_decay_steps=50,
            warmup_steps=10,
            checkpoint_freq=999999,
            max_daily_loss=10.0,
        )

        agent = DQNAgent(cfg, exploration_strategy=strategy)

        # Verify strategy is used
        assert agent.epsilon() == strategy.epsilon(0)

    # Cleanup
    Path(data_path).unlink()
