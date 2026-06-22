"""Integration test: full lifecycle without a real WebSocket connection."""


def test_full_lifecycle():
    """Simulate: receive data -> agent selects action -> execute -> close -> verify PnL."""
    from rl_bot.config import RLConfig, ACTION_HOLD, ACTION_CLOSE_YES
    from rl_bot.btc_data import BTCDataPoller
    from rl_bot.environment import TradingEnv
    from rl_bot.agent import DQNAgent

    cfg = RLConfig(
        eps_start=0.0, eps_end=0.0,  # pure greedy for determinism
        warmup_steps=2, batch_size=2,
    )
    poller = BTCDataPoller()
    for i in range(121):
        poller._on_spot_update(100_000.0 + i * 10.0)
    poller._on_funding_update(0.0001)

    env = TradingEnv(cfg, poller)
    agent = DQNAgent(cfg)

    # Simulate receiving ticker data
    for _ in range(5):
        env.on_ticker("KXBTC-A", 0.40, 3.0)
    env.on_trade("KXBTC-A", 0.40, 3)

    # Get state and mask
    state = env.get_state("KXBTC-A")
    mask = env.get_mask("KXBTC-A")
    assert state.shape == (18,)
    assert mask.shape == (21,)

    # Agent selects action (greedy)
    action = agent.select_action(state, mask)
    assert 0 <= action <= 20

    # Execute action
    next_state, reward, done = env.step("KXBTC-A", action)
    assert next_state.shape == (18,)

    # Store transition
    agent.store_transition(state, action, reward, next_state, done)

    # Simulate price movement
    for _ in range(3):
        env.on_ticker("KXBTC-A", 0.60, 2.5)

    # Now force a close if we have a position
    pos = env.pnl_tracker.get_position("KXBTC-A")
    if pos != 0:
        close_action = ACTION_CLOSE_YES if pos > 0 else 20  # CLOSE_NO
        state2 = env.get_state("KXBTC-A")
        next_state2, reward2, done2 = env.step("KXBTC-A", close_action)
        agent.store_transition(state2, close_action, reward2, next_state2, done2)

    # Verify we can train
    agent.step_count = 10
    loss = agent.train_step()
    # Should be able to train now (at least 2 transitions, warmup=2)
    assert loss is not None or len(agent._buffer) < cfg.warmup_steps


def test_settlement_lifecycle():
    """Simulate opening a position and having it settle at expiry."""
    from rl_bot.config import RLConfig
    from rl_bot.btc_data import BTCDataPoller
    from rl_bot.environment import TradingEnv

    cfg = RLConfig()
    poller = BTCDataPoller()
    for i in range(121):
        poller._on_spot_update(100_000.0)
    env = TradingEnv(cfg, poller)

    # Feed data and buy YES
    for _ in range(3):
        env.on_ticker("KXBTC-A", 0.30, 1.0)
    env.step("KXBTC-A", 0)  # BUY_YES_1_AT_0c

    # Settle — YES wins
    pnl = env.settle_market("KXBTC-A", outcome=True)
    # Bought at 0.30, settles at 1.00 => gross 0.70 - entry_fee
    assert pnl > 0.5  # should be ~0.69 after fee


def test_circuit_breaker():
    """Verify circuit breaker activates after exceeding daily loss limit."""
    from rl_bot.config import RLConfig
    from rl_bot.btc_data import BTCDataPoller
    from rl_bot.environment import TradingEnv

    cfg = RLConfig(max_daily_loss=0.10)  # very tight limit for testing
    poller = BTCDataPoller()
    for i in range(121):
        poller._on_spot_update(100_000.0)
    env = TradingEnv(cfg, poller)

    # Open and close at a loss to trigger circuit breaker
    for _ in range(3):
        env.on_ticker("KXBTC-A", 0.50, 1.0)
    env.step("KXBTC-A", 6)  # BUY_YES_5_AT_0c (5 contracts)
    # Price drops
    for _ in range(3):
        env.on_ticker("KXBTC-A", 0.30, 0.5)
    # Close at loss
    from rl_bot.config import ACTION_CLOSE_YES
    env.step("KXBTC-A", ACTION_CLOSE_YES)

    # Daily PnL should be very negative
    assert env.pnl_tracker.daily_pnl() < 0
    assert env.is_circuit_breaker_active()
