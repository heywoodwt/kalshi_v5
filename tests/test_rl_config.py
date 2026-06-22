import pytest


def test_rl_config_defaults():
    from rl_bot.config import RLConfig
    cfg = RLConfig()
    assert cfg.state_dim == 18
    assert cfg.n_actions == 21
    assert cfg.hidden_dim == 128
    assert cfg.gamma == 0.99
    assert cfg.max_position_per_market == 5
    assert cfg.max_total_markets == 10
    assert cfg.paper_trading is True
    assert cfg.maker_fee_rate == 0.0175
    assert cfg.sizes == (1, 3, 5)
    assert cfg.offsets == (0.0, 0.02, 0.04)


def test_action_constants():
    from rl_bot.config import ACTION_HOLD, ACTION_CLOSE_YES, ACTION_CLOSE_NO
    assert ACTION_HOLD == 18
    assert ACTION_CLOSE_YES == 19
    assert ACTION_CLOSE_NO == 20


def test_decode_action_buy_yes():
    from rl_bot.config import decode_action
    # action 0 = direction=0(yes), size_idx=0(1 contract), offset_idx=0(0c)
    result = decode_action(0)
    assert result == ("yes", 1, 0.0)


def test_decode_action_buy_no():
    from rl_bot.config import decode_action
    # action 9 = direction=1(no), size_idx=0(1 contract), offset_idx=0(0c)
    result = decode_action(9)
    assert result == ("no", 1, 0.0)


def test_decode_action_buy_yes_max():
    from rl_bot.config import decode_action
    # action 8 = direction=0(yes), size_idx=2(5 contracts), offset_idx=2(4c)
    result = decode_action(8)
    assert result == ("yes", 5, 0.04)


def test_decode_action_buy_no_max():
    from rl_bot.config import decode_action
    # action 17 = direction=1(no), size_idx=2(5 contracts), offset_idx=2(4c)
    result = decode_action(17)
    assert result == ("no", 5, 0.04)


def test_decode_action_hold():
    from rl_bot.config import decode_action
    assert decode_action(18) == "hold"


def test_decode_action_close_yes():
    from rl_bot.config import decode_action
    assert decode_action(19) == "close_yes"


def test_decode_action_close_no():
    from rl_bot.config import decode_action
    assert decode_action(20) == "close_no"


def test_decode_action_invalid():
    from rl_bot.config import decode_action
    with pytest.raises(ValueError):
        decode_action(21)
    with pytest.raises(ValueError):
        decode_action(-1)
