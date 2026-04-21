Integration complete. Here's a summary of what was done:

Files Created

┌───────────────────────────────┬──────────────────────────────────────────────────────────────────────────────┐                                                              
│             File              │                                   Purpose                                    │                                                            
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤                                                            
│ model/tft/__init__.py         │ Package marker                                                               │                                                            
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ model/tft/fees.py             │ Kalshi fee calculations (cents-based, self-contained)                        │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ model/tft/feature_engineer.py │ 17 features: log returns, volatility, momentum, RSI, z-scores, time encoding │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ model/tft/tft_dataset.py      │ TimeSeriesDataSet builder with train/val split                               │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ model/tft/tft_model.py        │ TFTPredictor class — train, predict, save/load                               │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ model/tft/data_fetcher.py     │ Exchange data fetchers (Binance/Coinbase/Kraken/Bybit) + synthetic data      │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ model/tft/train_tft.py        │ CLI training entry point                                                     │
└───────────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘

Files Modified

┌──────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
│           File           │                                         Changes                                          │
├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ model/model_interface.py │ Bridges TFT to v5: buffers WebSocket prices, runs periodic inference, caches predictions │
├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ .gitignore               │ Added checkpoints, lightning_logs, __pycache__                                           │
├──────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ requirements.txt         │ Added torch, pytorch-forecasting, lightning, polars, pyarrow, httpx                      │
└──────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘

Key Design Decisions

- Zero external dependencies on kalshi_v4 — all btc_signal_pipeline imports removed; fees inlined, data fetcher standalone
- Backward compatible — ModelInterface API unchanged; predict(ticker, features) -> float still works. Falls back to passthrough when no checkpoint exists
- Automatic buffering — model_interface.py accumulates price data from predict() calls and runs TFT inference every 30 seconds when 70+ observations are buffered
- Checkpoint = 3 files — .ckpt (weights), .json (hparams), _dataset.pt (TimeSeriesDataSet for reconstruction)

Usage

# Train with synthetic data
python -m model.tft.train_tft --synthetic --max-epochs 50

# Train from saved trade data
python -m model.tft.train_tft --data-file trades.parquet --enrich

# Run bot (auto-loads checkpoint if present)
python main.py

