import pytest
from rl_bot.mm_config import MMConfig


def test_config_auto_sets_demo_urls():
    """Test auto-configuration of demo API URLs."""
    cfg = MMConfig(api_environment="demo")

    assert cfg.api_base_url == "https://external-api.demo.kalshi.co/trade-api/v2"
    assert cfg.ws_url == "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"


def test_config_auto_sets_production_urls():
    """Test auto-configuration of production API URLs."""
    cfg = MMConfig(api_environment="production")

    assert cfg.api_base_url == "https://external-api.kalshi.com/trade-api/v2"
    assert cfg.ws_url == "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


def test_config_custom_url_override():
    """Test that custom URLs override auto-configuration."""
    custom_url = "https://custom-api.example.com/v2"
    custom_ws = "wss://custom-ws.example.com/v2"

    cfg = MMConfig(
        api_environment="demo",
        api_base_url=custom_url,
        ws_url=custom_ws,
    )

    assert cfg.api_base_url == custom_url
    assert cfg.ws_url == custom_ws
