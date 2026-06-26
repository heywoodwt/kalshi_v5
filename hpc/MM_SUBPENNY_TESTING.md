# Testing MM Subpenny on HPC

Quick guide to deploy and test the mm-subpenny version with orderbook integration on UVA Rivanna HPC.

## New Features Being Tested

1. **Subpenny Pricing**: +0.001 bid, -0.001 ask for queue priority
2. **Market Metadata Loader**: Validates tick sizes (deci_cent, linear_cent, tapered_deci_cent)
3. **16-Dimensional Observation Space**: Adds orderbook features (spread, imbalance, depth)
4. **Configuration System**: MMConfig with API endpoints and feature flags
5. **PPO Training**: Stable Baselines3 with checkpoint saving

## Prerequisites

- UVA VPN connected (if off-grounds)
- SSH access to `mtk9va@login.hpc.virginia.edu`
- Data files on HPC:
  - `output/rl_kalshi_trades_3mo.parquet` (trade data)
  - `output/rl_all_markets_3mo.parquet` (market metadata)

## Step 1: Deploy Code to HPC

From your **local machine** (macOS), on the `feature/mm-subpenny` branch:

```bash
# Make sure you're on the right branch
git branch --show-current  # should show: feature/mm-subpenny

# Deploy to HPC
bash hpc/deploy.sh
```

This syncs the code to `/scratch/mtk9va/kalshi_v5` on HPC.

## Step 2: Run Quick Test (1 hour)

SSH to HPC and run the test job:

```bash
# SSH to HPC
ssh mtk9va@login.hpc.virginia.edu

# Navigate to project
cd /scratch/mtk9va/kalshi_v5

# Create results directory
mkdir -p hpc/mm_results

# Submit test job (50k steps, KXBTC category)
sbatch hpc/test_mm_subpenny.slurm

# Monitor job status
squeue -u mtk9va

# Watch output (wait for job to start)
tail -f hpc/mm_results/mm_test-<job-id>.out
```

### What the Test Validates

- ✅ MMEnv initializes with 16-dim observation space
- ✅ Market metadata loads from parquet
- ✅ Subpenny pricing logic executes
- ✅ PPO training loop runs
- ✅ Checkpoints save correctly
- ✅ GPU allocation works

### Expected Output

```
=== Environment Info ===
Python 3.11.x
...

=== GPU Info ===
CUDA available: True
GPU count: 1
GPU 0: NVIDIA A100-SXM4-40GB
...

=== Starting MM Subpenny Test ===
Category: KXBTC
Timesteps: 50,000
...
Loading trades from output/rl_kalshi_trades_3mo.parquet
Loaded metadata for XXX markets
Created MM environment with 16-dim observation space
Subpenny pricing: enabled
...
```

### Test Success Criteria

1. Job completes in < 1 hour
2. Final checkpoint created: `rl_bot/mm_checkpoints/mm_subpenny_test_KXBTC_final.zip`
3. No Python errors in `.err` file
4. TensorBoard logs created: `rl_bot/mm_logs/mm_subpenny_test/`

## Step 3: Run Full Training (12 hours)

If the test passes, run the full training:

```bash
# From HPC
sbatch hpc/train_mm.slurm

# Monitor
squeue -u mtk9va
tail -f hpc/mm_results/mm_train_all-<job-id>.out
```

This trains on **all categories** with 500k steps each.

## Step 4: Check Results

After training completes:

```bash
# View checkpoints
ls -lh rl_bot/mm_checkpoints/

# Check logs
ls -lh rl_bot/mm_logs/

# Download results to local machine (from local terminal)
rsync -avz mtk9va@login.hpc.virginia.edu:/scratch/mtk9va/kalshi_v5/rl_bot/mm_checkpoints/ ./rl_bot/mm_checkpoints/
```

## Troubleshooting

### Job Fails with "No module named 'stable_baselines3'"

Install in conda environment:
```bash
conda activate kalshi_rl
pip install stable-baselines3[extra]
```

### Job Fails with "No such file: output/rl_kalshi_trades_3mo.parquet"

Ensure data files are on HPC:
```bash
ls -lh output/*.parquet
```

If missing, transfer from local:
```bash
# From local machine
scp output/rl_kalshi_trades_3mo.parquet mtk9va@login.hpc.virginia.edu:/scratch/mtk9va/kalshi_v5/output/
scp output/rl_all_markets_3mo.parquet mtk9va@login.hpc.virginia.edu:/scratch/mtk9va/kalshi_v5/output/
```

### Job Pending in Queue

Check allocation status:
```bash
allocations
squeue -u mtk9va
```

### Out of Memory

Reduce batch size in `rl_bot/mm_train.py`:
```python
batch_size=32,  # was 64
```

## Testing Without Subpenny

To compare performance with/without subpenny:

```bash
# Test without subpenny pricing
python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_3mo.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category KXBTC \
    --timesteps 50000 \
    --no-subpenny \
    --run-name mm_no_subpenny
```

## Next Steps

After successful HPC testing:

1. Compare checkpoints: subpenny vs. baseline
2. Evaluate on test set
3. Backtest on recent data
4. Deploy to production with API mode
