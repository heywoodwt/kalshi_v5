#!/bin/bash
#SBATCH -J mm_submit_all
#SBATCH -o hpc/mm_results/mm_submit_all-%A.out
#SBATCH -e hpc/mm_results/mm_submit_all-%A.err
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -A sds_capstone_atashman

# Master script to submit all per-category MM PPO training jobs.
# This script simply loops through hpc/mm_jobs/*.slurm and submits each.
#
# Usage:
#   1. Generate per-category scripts: python hpc/generate_mm_slurm.py
#   2. Submit all jobs:               sbatch hpc/train_mm_all.slurm
#   3. Monitor jobs:                  squeue -u mtk9va

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

JOBS_DIR="hpc/mm_jobs"

# Check if jobs directory exists and has .slurm files
if [ ! -d "$JOBS_DIR" ]; then
    echo "ERROR: Jobs directory not found: $JOBS_DIR"
    echo "Run 'python hpc/generate_mm_slurm.py' first to generate job scripts."
    exit 1
fi

# Count job scripts
n_jobs=$(ls "$JOBS_DIR"/*.slurm 2>/dev/null | wc -l)
if [ "$n_jobs" -eq 0 ]; then
    echo "ERROR: No .slurm files found in $JOBS_DIR"
    echo "Run 'python hpc/generate_mm_slurm.py' first to generate job scripts."
    exit 1
fi

echo "=== Submitting MM PPO Training Jobs ==="
echo "Jobs directory: $JOBS_DIR"
echo "Number of jobs: $n_jobs"
echo ""

# Submit each job and collect job IDs
job_ids=()
for script in "$JOBS_DIR"/*.slurm; do
    job_name=$(basename "$script" .slurm)
    echo "Submitting $job_name..."

    # Submit job and capture job ID
    job_id=$(sbatch "$script" | awk '{print $NF}')
    job_ids+=("$job_id")

    echo "  → Job ID: $job_id"
done

echo ""
echo "=== Submission Complete ==="
echo "Jobs submitted: ${#job_ids[@]}"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor jobs:"
echo "  squeue -u mtk9va"
echo ""
echo "Check results:"
echo "  ls -lh hpc/mm_results/"
echo "  ls -lh rl_bot/mm_models/"
