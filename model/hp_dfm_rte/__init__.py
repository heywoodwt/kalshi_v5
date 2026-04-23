"""HP-DFM-RTE model package: HP Filter + Dynamic Factor Model - Real-Time Ensemble."""

from .model_engine import TickerForecast, HPCycleDFMEngine, EWMAEngine, make_engine
from .signal_gen import SignalGenerator, Signal, SignalDirection
from .config import PipelineConfig
from .accumulator import TradeAccumulator