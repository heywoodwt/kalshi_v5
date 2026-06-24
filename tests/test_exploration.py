import pytest
import numpy as np
from rl_bot.exploration import ExplorationStrategy


def test_base_class_cannot_be_instantiated():
    """ExplorationStrategy is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        ExplorationStrategy({})
