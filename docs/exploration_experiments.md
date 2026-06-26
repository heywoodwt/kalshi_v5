# Exploration Strategy Experiments (2026-06-24)

## Background

The RL trading agent's original epsilon-greedy exploration used a slow decay (~75% decay rate) that barely reached epsilon=0.92 by end of training. Each random exploratory trade costs ~$0.05-$0.12 in fees, burning through the $500 circuit breaker budget. This experiment tested 6 alternative exploration strategies, each run with and without the `--prioritize-held` flag (12 runs total).

## Strategies Tested

| Strategy | Decay Type | Config | How It Explores |
|----------|-----------|--------|-----------------|
| `fast_linear` | Linear: 0.5 → 0.05 over `n_rows/10` steps | eps_start=0.5, eps_end=0.05 | Standard epsilon-greedy, random valid action |
| `exponential` | Exponential: 0.8 × 0.999^step | eps_start=0.8, decay_rate=0.999 | Standard epsilon-greedy |
| `logarithmic` | Log curve: 0.8 → 0.05 over `n_rows/2` steps | eps_start=0.8 | Standard epsilon-greedy |
| `episode` | Per-episode exponential: 0.8 × 0.99^episode | eps_start=0.8, decay_rate=0.99 | Standard epsilon-greedy |
| `action_local` | Linear (same as fast_linear) | eps_start=0.6 | Explores near greedy action (same direction YES/NO, vary size/offset) |
| `parameter_noise` | No epsilon (always greedy) | noise_std: 1.0 → 0.1 | Gaussian noise added to Q-values before argmax |

## Results Ranked by Final PnL

| Rank | Strategy | Held | Final PnL | Buy Trades | Closes | Eps Floor Step |
|------|----------|------|-----------|------------|--------|----------------|
| 1 | parameter_noise | no | **-$3.77** | 449 | 12 | 0 |
| 2 | action_local | no | -$3.79 | 450 | 0 | 4,267 |
| 3 | logarithmic | no | -$4.70 | 525 | 12 | 10,551 |
| 4 | exponential | no | -$5.34 | 346 | 9 | 2,751 |
| 5 | fast_linear | no | -$5.49 | 472 | 12 | 2,133 |
| 6 | episode | no | -$18.18 | 1,888 | 122 | never |
| 7 | fast_linear | yes | -$303.28 | 46,157 | 1,317 | 2,133 |
| 8 | logarithmic | yes | -$350.10 | 49,584 | 1,489 | 10,551 |
| 9 | exponential | yes | -$354.39 | 48,279 | 717 | 2,751 |
| 10 | action_local | yes | -$393.42 | 52,287 | 617 | 4,267 |
| 11 | parameter_noise | yes | -$421.93 | 57,758 | 331 | 0 |
| 12 | episode | yes | -$500.31 | 41,966 | 2,252 | never |

## Key Findings

### 1. `--prioritize-held` is catastrophic

Every single `no_held` variant outperformed its `held` counterpart by $300-$480. The held flag forces the agent to re-evaluate positions it already holds, generating 40,000-57,000 buy trades vs 350-525 without it. Each trade incurs fees, so the held variants bleed money on transaction costs regardless of strategy quality.

| Strategy | held PnL | no_held PnL | Delta |
|----------|----------|-------------|-------|
| fast_linear | -$303.28 | -$5.49 | -$297.80 |
| exponential | -$354.39 | -$5.34 | -$349.05 |
| logarithmic | -$350.10 | -$4.70 | -$345.40 |
| episode | -$500.31 | -$18.18 | -$482.13 |
| action_local | -$393.42 | -$3.79 | -$389.63 |
| parameter_noise | -$421.93 | -$3.77 | -$418.16 |

### 2. Without held, the agent learns to mostly hold (which is correct)

The `no_held` variants made 346-525 buy trades out of 21,377 steps (~2% trade rate). This means the agent quickly learned that most trading opportunities are negative EV after fees. The best behavior is overwhelmingly HOLD, with selective trades only when the edge is sufficient.

### 3. Episode-based decay is broken with 1 epoch

The `episode` strategy decays epsilon per episode, not per step. With only 1 epoch of training, epsilon stayed at 0.800 the entire run — the agent explored randomly 80% of the time from start to finish. This is why episode_no_held lost $18.18 (4x worse than other no_held variants) and episode_held hit the $500 circuit breaker.

### 4. No_held strategies are nearly equivalent

Among the 5 working `no_held` strategies, final PnL ranges from -$3.77 to -$5.49. The differences are within noise. All converge to the same behavior: hold almost everything, trade rarely. The exploration strategy matters far less than the held/no_held decision.

### 5. Win rates are extremely low across the board

| Variant | Win Rate | Avg Win | Avg Loss |
|---------|----------|---------|----------|
| no_held strategies | 0-5.4% | $0.00-$0.58 | -$0.06 to -$0.12 |
| held strategies | 0.1-0.3% | $0.21-$0.59 | -$0.05 to -$0.05 |

The agent wins on very few trades. The fee structure ($0.05+ per trade) means most trades are negative EV. The no_held variants survive by avoiding trades; the held variants are forced to trade and bleed out.

### 6. Among held variants, fast_linear loses least

If forced to use `--prioritize-held`, `fast_linear` (-$303) outperforms the rest because it reaches the epsilon floor fastest (step 2,133), cutting off random exploration early. Slower decay = more random trades = more fee bleed.

## Infrastructure

- **HPC**: UVA Rivanna, A100 GPU, 4 CPU cores, 32GB RAM, 2h wall time
- **Data**: `rl_kalshi_trades_3mo.parquet` (21,377 1-min bars, ~3 months of BTC markets)
- **Circuit breaker**: $500 max loss
- **All runs**: 1 epoch, speed=0 (no delay)
- **Runtime**: ~23 min per experiment

## Files

- Strategies: `rl_bot/exploration.py` (6 classes + ABC base)
- Strategy factory + CLI: `rl_bot/replay.py` (`create_exploration_strategy()`, `--strategy` flag)
- Agent integration: `rl_bot/agent.py` (`exploration_strategy` parameter)
- SLURM generator: `hpc/generate_exploration_slurm.py`
- SLURM scripts: `hpc/exp_<strategy>_<held|no_held>.slurm` (12 scripts)
- Trade logs: `output/rl_replay_exp_<name>_trades.csv`
- Tests: `tests/test_exploration.py`, `tests/test_exploration_replay.py` (18 tests)

## Reproducing

```bash
# Generate SLURM scripts
python hpc/generate_exploration_slurm.py

# Deploy to HPC
bash hpc/deploy.sh

# Submit all 12 experiments
ssh mtk9va@login.hpc.virginia.edu
cd /scratch/mtk9va/kalshi_v5
for f in hpc/exp_*.slurm; do sbatch "$f"; done

# Monitor
squeue -u mtk9va

# Download results
rsync -avz mtk9va@login.hpc.virginia.edu:'/scratch/mtk9va/kalshi_v5/output/rl_replay_exp_*_trades.csv' output/
```

## Next Steps

1. **Drop `--prioritize-held`** — it's strictly worse across every strategy
2. **Fix episode-based strategy** — either run multi-epoch or switch to step-based decay
3. **Investigate fee structure** — the agent's optimal policy is "almost never trade," suggesting fees dominate any possible alpha. Consider: lowering simulated fees, adding a fee-aware reward shaping, or targeting only high-edge opportunities
4. **Multi-epoch runs** — current runs use 1 epoch. Multiple passes over the data may help the agent learn better when to trade vs hold
