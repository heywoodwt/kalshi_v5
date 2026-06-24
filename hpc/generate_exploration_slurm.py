#!/usr/bin/env python3
"""Generate SLURM job scripts for all exploration strategy experiments.

Creates 12 .slurm files (6 strategies × 2 prioritize-held variants).
Each script runs replay.py with unique --strategy and --run-name flags.

Usage:
    python hpc/generate_exploration_slurm.py

Output:
    hpc/exp_<strategy>_<held|no_held>.slurm (12 files)
"""
import os
from pathlib import Path

# 6 exploration strategies
STRATEGIES = [
    "fast_linear",
    "exponential",
    "logarithmic",
    "episode",
    "action_local",
    "parameter_noise",
]

# SLURM template with placeholders
TEMPLATE = """#!/bin/bash
#SBATCH -J {job_name}              # Job name
#SBATCH -o {output_dir}/{job_name}-%A.out  # stdout log
#SBATCH -e {output_dir}/{job_name}-%A.err  # stderr log
#SBATCH -p gpu                     # GPU partition
#SBATCH --gres=gpu:a100:1          # 1x A100 GPU
#SBATCH -c 4                       # 4 CPU cores
#SBATCH --mem=32G                  # 32 GB RAM
#SBATCH -t 02:00:00                # 2 hour wall time
#SBATCH -A sds_capstone_atashman   # SU allocation group

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

# Load conda environment
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Experiment: {experiment_name} ==="
echo "Strategy: {strategy}"
echo "Prioritize Held: {prioritize_label}"
echo "Circuit Breaker: $500"
echo ""

# Run replay with exploration strategy
python -m rl_bot.replay \\
    --data output/rl_kalshi_trades_3mo.parquet \\
    --speed 0 \\
    --max-loss 500 \\
    --strategy {strategy} \\
    {prioritize_flag}\\
    --run-name {run_name}

echo ""
echo "=== Experiment {job_name} complete ==="
echo "Checkpoints: rl_bot/checkpoints_{run_name}/"
echo "Trade log: output/rl_trades_{run_name}.csv"
"""


def main():
    output_dir = "hpc/exploration_exps"
    script_dir = Path("hpc")

    # Create output directory for logs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    generated_files = []

    for strategy in STRATEGIES:
        for prioritize in [False, True]:
            # Naming convention
            held_suffix = "held" if prioritize else "no_held"
            job_name = f"exp_{strategy}_{held_suffix}"
            run_name = job_name
            prioritize_label = "Yes" if prioritize else "No"
            prioritize_flag = "--prioritize-held \\" if prioritize else ""

            # Fill template
            script_content = TEMPLATE.format(
                job_name=job_name,
                output_dir=output_dir,
                experiment_name=job_name.replace("_", " ").title(),
                strategy=strategy,
                prioritize_label=prioritize_label,
                prioritize_flag=prioritize_flag,
                run_name=run_name,
            )

            # Write script
            output_path = script_dir / f"{job_name}.slurm"
            output_path.write_text(script_content)

            # Make executable
            os.chmod(output_path, 0o755)

            generated_files.append(output_path.name)

    print(f"Generated {len(generated_files)} SLURM scripts in {script_dir}/")
    print("\nTo submit all experiments on HPC:")
    print("  cd /scratch/mtk9va/kalshi_v5")
    print("  for f in hpc/exp_*.slurm; do sbatch \"$f\"; done")
    print("\nTo monitor:")
    print("  squeue -u mtk9va")
    print("  tail -f hpc/exploration_exps/exp_*-%A.out")


if __name__ == "__main__":
    main()
