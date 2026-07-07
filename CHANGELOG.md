# Changelog

## 2026-07-07 (S3-refresh retrain results — SLURM 16816184)

Retrained on the 8-day S3 window (6.6M trades, split 2026-07-06) with the
Phase 3 sim. Out-of-sample (July 6-7 holdout), all checkpoints verified 20-dim
and pulled local (July-1 finals kept as *.jul01.zip backups):

- KXBTCD  +916.91 / 576 eps (398 traded, W/L 194/204). Caveat: 60 short
  (1-2 step) episodes contribute +439.72 (~48%) — thin-strike flatten marks,
  treat the ex-artifact ~+477 as the honest figure. Worst episode -11.06.
- KXWCGAME +64.82 / 28 eps (18 traded, 11W/7L). Cleanest result: zero
  artifact episodes, worst only -2.12.
- KXAAAGASD +6.17 / 29 eps — POSITIVE now (July-1 model, trained on April
  data, evaled -2.37). Artifacts NET -22.10 here, so genuine spread capture
  ~+28. Candidate to re-add to the lowvol whitelist.
- KXAAAGASM (421 trades in window) and KXADP (9) not retrained.

Noted: the category prefix filter also matches KXWCGAMEGOALS tickers in
KXWCGAME training/eval (4/28 eval episodes, $0 PnL impact — benign for now).

## 2026-07-07 (profile-guided speed pass)

Profiled both hot paths before touching anything. Findings and fixes:

- **Live WS callback (the win that matters)**: `model.predict` ran on EVERY
  book tick even though `_execute_action`'s 1s throttle then discarded the
  result — 72% of callback time wasted. The callback now folds the message
  into the LiveBook (always — a stale book poisons later decisions), then
  peeks the throttle BEFORE obs build + predict. Throttled ticks (the common
  case): 89us -> 0.9us, ~99x. Full quote path: 89us -> 74us.
- **Risk check off the per-tick path**: `_check_risk_limits` scanned all open
  positions and rebuilt a 310-entry set per message; now runs only for
  unthrottled ticks, with the active-ticker set cached.
- **`scale_action` de-numpy'd**: np.clip on scalars costs ~40x a min/max
  pair; runs per env step and per live quote. Pure Python now (also directly
  Rust-translatable). Training env: 26.4k -> 40.2k steps/s (+52%).
- **Obs builders**: scalar np.clip -> min/max; live obs clip bounds hoisted
  to module constants (two array allocations per tick eliminated).

Deliberately NOT optimized (measured, not worth it): preprocess_mm_data
(0.07s / 140k trades, already vectorized), SB3 predict internals (0.061ms,
model is tiny), REST startup loops (bound by Kalshi's ~10 req/s rate limit,
not by code), PPO training throughput (GPU-bound; env is ~2.4% of budget).

## 2026-07-01 (validation-week launch: capital guards + positions parser fix)

First live run with the Phase 1-3 fixes confirmed post-only works: 9/9 fills
maker, ZERO taker. Three launch problems found and fixed (all tested):

- **Positions API parser was reading a nonexistent key**: Kalshi V2 nests
  positions under `market_positions` with `position_fp` fixed-point strings;
  the code read `response["positions"]` -> always empty -> position sync and
  the startup order-optimizer had been silent no-ops. Fixed in `_sync_positions`
  and `initialize()` (accepts both shapes).
- **Capital guards** (`live_trader_v2.py` + MMConfig): a two-sided 1-lot quote
  locks ~$1 collateral, so quoting 310 markets with ~$95 produced 4,900+
  insufficient_balance rejections in 6 minutes. Now: 60s global quoting pause
  on an insufficient_balance rejection (`balance_backoff_s`), a global cap of
  60 concurrent resting orders (`max_open_orders`), and a quote price band of
  [0.05, 0.95] mid (`quote_band_lo/hi`) — 97c contracts tie up 97c to earn
  pennies near settlement.
- **Fractional fills ignored**: Kalshi reports count_fp like 0.5; int() floored
  them to size-0 "fills" that bumped fee/taker counters while changing nothing.
- **Per-tick Action log demoted to debug** (was 45k lines / 6 min at INFO).

Account state discovered at launch: $0.0008 free cash, $85.82 locked in 117
positions left by the pre-fix era. ~$33.51 settles this week (frees itself);
~$32 of 2027-dated shorts (KXAISTREAMSERIES-27, KXARREST-27JAN, KXAISPIKE-27B,
...) is UNCLOSABLE — books are literally empty, no counterparty exists. Trader
left running; quoting begins automatically as settlements release collateral.

## 2026-07-01 (Phase 3 results — retrained models evaluated, whitelist finalized)

SLURM array 16705279 trained all four categories (500k steps each). Discovery:
`rl_kalshi_trades_3mo.parquet` is mislabeled — it only spans 2026-04-18..04-25,
so the 06-15 eval split had no test data. Re-evaluated the three affected models
locally on the S3 parquet (June 29-Jul 1) — a genuine 2-months-forward holdout.

Out-of-sample scorecard (total PnL over eval episodes, Phase 3 fee model):
- KXBTCD  +479.18 (262 eps) — from the earlier 16691732 job, 7/01 split
- KXWCGAME +121.26 (43 eps, median +0.60, worst episode only -2.16) — strongest
- KXAAAGASM +21.38 (32 eps) — positive but variance-dominated (thin-market
  1-step episodes swing +/-10 on flatten marks); treat with caution at scale
- KXADP unevaluable (8 trades in holdout, 0 fills) — kept in the whitelist on
  the live markout evidence (+0.24 net over 10 fills, ~zero adverse selection)
- KXAAAGASD -2.37 (4/26 winning eps) — REMOVED from `live_config_lowvol.py`

All four whitelist checkpoints verified 20-dim and pulled to
`rl_bot/mm_checkpoints/realistic_20dim_{CAT}_final.zip`. Restarting the live
trader activates them. Caveat for expectations: sim fills reach 20 contracts
per window (max_inventory) while live quotes are 1-lot with re-quote throttle,
so live PnL will be a small fraction of these eval totals.

## 2026-07-01 (Phase 3 — close the remaining sim-live gap, retrain low-vol models)

Sim fixes (all tested in `tests/test_mm_bias_fixes.py`, 12 tests):

- **Through-fills execute at OUR quote price** (`mm_env.py`): a resting order
  fills at its own limit price (price-time priority), but the sim filled at the
  print price — buying below our bid / selling above our ask — crediting phantom
  edge on every through-fill.
- **Fees at live order granularity** (`reward.py` `fee_at_quote_size()`): Kalshi
  ceils each ORDER's fee to the next cent; live quotes are quote_size(=1)-lot
  orders, so an N-contract sim fill is really N orders at 1c each (0.50 price) —
  2.3x what the old single-ceil fee charged. Used by `_fill_buy/_fill_sell` and
  the episode-end flatten.
- **Episode-end flatten pays the TAKER rate** (`mm_env.py`): crossing the spread
  to exit is a taker execution; it was charged the maker rate.
- **`taker_fill_prob` default 0.945 -> 0.10** (`mm_config.py`): 94.5% taker was
  the PRE-fix live pathology. With post-only quoting only stop-loss/expiry exits
  cross. Re-measure from the hourly summary each deployment week.

Live obs parity (training semantics replicated exactly; `tests/test_live_book.py`):

- **[9]/[16] from real trade prints**: subscribed the WS "trade" channel;
  `trade_window_features()` counts prints/50 and contract-weighted taker-side
  flow over 60s windows. The old code used book-depth proxies — [9] measured
  liquidity (not activity) and [16] duplicated [8] book imbalance.
- **[14] realized PnL is per-ticker** (like a training episode), no longer the
  account-wide daily total leaking other markets' PnL into every obs.
- **[12] tte capped at 24h**: training episodes start at 24h (tte_log <= 3.22);
  far-dated live markets were clipping at 4.0, out of distribution.

HPC retraining (Phase 2 whitelist completion):

- **`hpc/train_mm_lowvol.slurm`** (array job 0-3) + **`hpc/deploy_lowvol.sh`**:
  trains realistic_20dim models for KXWCGAME (S3 3-day data, 47k trades, split
  2026-07-01) and KXADP/KXAAAGASM/KXAAAGASD (3-month data — these markets only
  print 1k-9k trades even over 3 months, split 2026-06-15). Checkpoints land as
  `realistic_20dim_{CAT}_final.zip`, which the live trader's dim check accepts
  automatically. NOT yet submitted — HPC SSH needs interactive auth; run
  `bash hpc/deploy_lowvol.sh --submit`.

## 2026-07-01 (Phase 2 — deploy only where the edge is proven)

- **`rl_bot/live_config_lowvol.py`** (new, now the DEFAULT config): whitelist of the
  five low-vol categories with demonstrated/plausible live edge — KXAAAGASM,
  KXAAAGASD, KXWCGAME, KXADP, KXBTCD (all vol < 0.05). KXBTC (0.070) and KXMLBGAME
  (0.185) are explicitly excluded: the markout analysis shows adverse selection eats
  the spread there. Tight validation-week limits (max_daily_loss $5, 1-lot quotes,
  max_inventory 5). `checkpoint_prefix="realistic_20dim"`.

- **Model/obs compatibility guard** (`live_trader_v2.py` `initialize()`): every
  loaded checkpoint's observation space must be (20,) or the category is DISABLED
  with a warning. Audit found ALL previously deployed checkpoints (june_*, mm_*)
  are 16-dim — they were being fed 20-dim observations live. Currently only
  `realistic_20dim_KXBTCD_final.zip` qualifies; the other four categories activate
  automatically once their retrained 20-dim models land (Phase 3 / HPC).

- **Vol filter timescale fixed** (`rl_bot/live_book.py` `sample_mid()`): mid-price
  history is now sampled on 60-second windows (last entry refreshed in place
  between boundaries) instead of appended per book tick. Momentum, velocity,
  realized vol, and the 0.05 vol-filter threshold were all calibrated on 60s
  training windows — per-tick appends ran them ~60x too fast, making the filter
  and those obs features meaningless. Tested in `tests/test_live_book.py`.

- **Deployment gate monitoring**: `_process_fill` tracks taker/maker fill counts and
  total fees; the hourly summary prints the taker fraction with a PASS/FAIL gate at
  10% (a working MM bot should be ~0% taker). Weekly scale-up gates documented in
  the config: taker < 10%, fees at maker schedule, live PnL >= 0.

## 2026-07-01 (Phase 1 live-executor fixes — why live lost money while sim won)

Live diagnosis found the bot was structurally a *taker* (94.5% of fills): it parsed
the orderbook from the wrong end, never used post-only orders, and quoted 1-lot
spreads that Kalshi's ceil'd fees fully consumed. Fixes, all tested in
`tests/test_live_book.py` (24 tests) + existing MM suites (27 total passing):

- **`rl_bot/live_book.py`** (new): canonical `LiveBook` orderbook. Kalshi delivers
  three conflicting level orderings (REST: ascending best-LAST; WS snapshots; WS
  deltas) and `live_trader_v2.py` read index `[0]` (the *worst* level) as best —
  corrupting every mid, quote center, depth feature, and stop-loss decision.
  LiveBook stores sides as price-keyed dicts so ordering never matters; best bid =
  max(yes), best ask = 1 - max(no). Also: `clamp_quotes()` (quotes can never cross
  the touch) and `quote_edge_ok()` (refuse spreads that round-trip ceil'd fees eat —
  at size=1 near 0.50 a round trip costs $0.02 regardless of the 1.75% nominal rate).

- **`rl_bot/live_trader_v2.py`**: all three book consumers (WS callback, REST
  polling, exit loop) now use LiveBook; the old delta handler with inconsistent
  yes-descending/no-ascending sorts is deleted. Quotes are placed `post_only=True`
  (exchange rejects crossing instead of filling as taker at the 7% rate) after
  clamping and the fee gate. Re-quote only when the desired price moves >= 1 tick —
  the old cancel-replace-every-tick + 30s stale cancel forfeited all queue priority.
  Fill tracking moved from per-order status polling to `GET /portfolio/fills`
  reconciliation (one call/5s): books the *actual* taker/maker fee per fill on both
  legs (old code charged maker fee on closing legs only). Stop-loss threshold is now
  max(5c, 2x spread) computed from the corrected mid, and exits go out as true IOC.
  One-instance flock (`~/.kalshi_mm_live.lock`) + per-config log files stop the
  duplicate-order bursts seen in the 6/30 logs (multiple processes, shared log).

- **`rl_bot/kalshi_api.py`**: `place_limit_order()` gained `post_only` and
  `time_in_force` passthrough (API already supported both in `create_order`).

- **`rl_bot/mm_config.py`**: new `min_quote_edge` (default $0.01/contract) — required
  profit beyond round-trip fees before quoting.

- **Entry-price tracking fixed for shorts** (`_process_fill`): opening a short from
  flat now records an entry price (the old code only did so for longs, so short
  positions were invisible to the stop-loss loop and covered with wrong PnL), and
  realized PnL on flattening fills uses the pre-fill entry price (previously the
  entry was popped first, zeroing the realized PnL of every full close).

## 2026-07-01 (S3 data pull + HPC KXBTCD training)

- **`consolidate_s3_data.py`**: new ETL script. Pulls all trades/orderbooks from
  the `kalshi-data-prod` S3 bucket (us-east-2, 96 hourly parquets, 2026-06-29..07-01)
  and consolidates them into the training format: `output/rl_kalshi_trades_s3.parquet`
  (1,591,079 trades — schema already matches `preprocess_mm_data`) and
  `output/s3_orderbooks.parquet` (274,562 book snapshots, up from the prior 831).
  Orderbook columns are renamed to mm_env's schema (implied_spread→spread,
  yes_best_size→yes_size, etc.) and a depth-based `imbalance` is derived. This
  resolves the "collect more orderbook data" item — training windows now get real
  book features instead of hardcoded defaults.

- **`hpc/train_mm_kxbtcd.slurm`** + **`hpc/deploy_kxbtcd.sh`**: new HPC job to train
  a 20-dim bias-corrected PPO model on KXBTCD (BTC daily, low-vol / MM-friendly).
  Temporal split at 2026-07-01 (80.8k train / 59.4k test KXBTCD trades). Targeted
  deploy pushes code + the two S3 parquets only (markets metadata already on HPC),
  avoiding the 900MB of stale parquets in output/. Submitted as Rivanna job
  16691732 (A100, sds_capstone_atashman allocation).

## 2026-07-01 (sim-vs-live bias fixes)

Implemented the fixes from `docs/sim_vs_live_bias.md` (sections 4 and 6, Phase 1-2 plus
the live volatility filter from Phase 3). All changes tested in `tests/test_mm_bias_fixes.py`.

- **Post-fill inventory marking** (`rl_bot/mm_env.py` `step()`): the reward now marks
  inventory at the *next* window's VWAP instead of the current window's VWAP. The print
  that fills us is informed flow, so the mid drifts against the new position; marking
  pre-fill hid that adverse-selection cost and upward-biased sim PnL (+9,744 eval vs
  -$0.31 live on KXBTC). The observation still uses the current mid — no look-ahead
  leaks into what the agent sees, only the reward label is corrected.

- **Taker-fee modeling** (`rl_bot/mm_env.py`, `rl_bot/reward.py`, `rl_bot/mm_config.py`):
  added `compute_taker_fee()` (same variance formula, 0.07 rate) and new config knobs
  `taker_fee_rate=0.07` / `taker_fill_prob=0.945`. Live data shows 94.5% of fills are
  taker, so the sim now charges taker fees on that fraction of fills instead of assuming
  100% maker (fee drag was understated ~4x).

- **Through-fill haircut 0.50 -> 0.33** (`rl_bot/mm_config.py`): assume we capture ~1/3
  of through volume (realistic FIFO queue position), down from 1/2.

- **20th observation dimension — realized volatility** (`rl_bot/mm_env.py`,
  `rl_bot/live_trader_v2.py`): `std(mid_history[-20:]) / 0.05`, clipped to [0, 1],
  identical formula in sim and live (`realized_vol()` helper). Lets one policy learn to
  widen spreads in volatile regimes instead of needing per-regime models.

- **Live volatility filter** (`rl_bot/live_trader_v2.py`): skip quoting entirely when
  rolling mid vol exceeds `vol_filter_threshold=0.05` — the empirical boundary between
  MM-friendly (gas/ADP/BTCD) and MM-hostile (KXBTC/MLBGAME) categories. Mid history
  retention raised 10 -> 20 samples for the vol window.

- **Calibration script timestamp fix** (`analyze_sim_vs_live.py`): the Kalshi API
  returns fill `ts` in epoch seconds, not milliseconds — normalize before matching
  against trade prints (previously 0 fills matched).

- **Phase-1 measurement run** (168h fills, 30 matched against June trades):
  low-vol categories confirm the doc's prediction — KXADP/KXAAAGASD show
  adverse_per_contract = 0.0000 (sim trustworthy there), KXAAAGASM shows 0.0018-0.0064
  (small but real, grows with horizon). No KXBTC fills in the window (not deployed).
  Results: `output/calibration_latest.csv`. More fills needed for statistical power
  (~100/category per the doc's power analysis).

- **BREAKING**: observation space is now 20-dim. Existing 19-dim checkpoints are
  incompatible with `live_trader_v2.py` — retrain before next deployment (retraining
  is required anyway since the reward label changed). Stale 16-dim assertions in
  `tests/test_mm_observation_space.py` / `tests/test_mm_full_integration.py` updated.

## 2026-07-01

- Add `analyze_sim_vs_live.py`: sim-vs-live calibration via markout analysis.
  Measures the adverse-selection bias baked into the MM simulator (`rl_bot/mm_env.py`),
  which credits fills at the contemporaneous mid and ignores the price drift that
  follows a fill. For each real fill it computes the simulator's credited edge
  (`entry_edge`), the edge that survives `dt` seconds later (`markout`), and the gap
  between them (`adverse` = adverse selection) per category and horizon. Fills come
  from the Kalshi API (or a cached parquet); mids come from the trades parquet.
  Output: `output/sim_vs_live_calibration.csv` plus a printed summary. Verified
  end-to-end on 32 local-log fills against `output/rl_kalshi_trades_june.parquet`.

- Expand `docs/sim_vs_live_bias.md` with detailed statistical explanations for MSc
  data scientists: formal problem formulation (section 0), hypothesis testing and
  power analysis (section 3.6), parametric adverse-selection models (section 3.5),
  why standard ML techniques can't fix a biased label (section 4.9), and econometric
  perspective via omitted variable bias (section 4.10). Includes practical guidance:
  sample size requirements, effect sizes, cross-validation approaches.
