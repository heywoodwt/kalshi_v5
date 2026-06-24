"""Compare exploration strategy experiment results.

Parses all experiment CSVs, computes summary metrics, generates comparison
tables and plots for PnL, epsilon trajectories, trade frequency, and cost
efficiency.

Usage:
    python analysis/compare_exploration.py \
        --csv-dir output \
        --output-dir output/exploration_analysis
"""
import argparse
from pathlib import Path
import polars as pl

# Threshold for filtering near-zero rewards to identify actual trades.
# Rewards below this magnitude are treated as no-op/holding actions.
TRADE_REWARD_THRESHOLD = 1e-9

# Epsilon exploration floor value with small tolerance.
# Target epsilon floor is 0.05, but we use 0.051 to account for floating-point
# precision and ensure we capture all rows that have reached the floor.
EPSILON_FLOOR_THRESHOLD = 0.051


def load_experiment_results(csv_dir: Path) -> dict[str, dict]:
    """Load all experiment CSVs and compute summary metrics.

    Args:
        csv_dir: Directory containing rl_trades_*.csv files

    Returns:
        Dict mapping experiment name to metrics:
        {
            "exp_fast_linear_held": {
                "df": pl.DataFrame,
                "total_pnl": -45.23,
                "total_trades": 342,
                "avg_cost_per_trade": -0.132,
                "final_epsilon": 0.05,
                "steps_until_eps_05": 1234,
                "trade_timeline": [12, 45, 67, ...],  # step numbers
                "pnl_curve": [(0, 0.0), (12, -1.2), ...],  # (step, cumulative_pnl)
                "epsilon_curve": [(0, 0.5), (100, 0.45), ...],
            },
            ...
        }
    """
    pattern = "rl_trades_exp_*.csv"
    csv_files = sorted(csv_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No experiment CSVs found matching {csv_dir}/{pattern}"
        )

    results = {}
    for csv_path in csv_files:
        # Extract experiment name from filename
        # e.g., "rl_trades_exp_fast_linear_held.csv" -> "exp_fast_linear_held"
        exp_name = csv_path.stem.replace("rl_trades_", "")

        try:
            # Load CSV with error handling
            df = pl.read_csv(csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Failed to read CSV {csv_path}: {e}"
            ) from e

        # Verify required columns exist
        required_cols = ["cumulative_pnl", "reward", "epsilon", "step"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {missing_cols}"
            )

        try:
            # Compute metrics
            total_pnl = float(df["cumulative_pnl"].tail(1).item())

            # Trades = rows where reward changed (abs(reward) > threshold)
            total_trades = df.filter(
                pl.col("reward").abs() > TRADE_REWARD_THRESHOLD
            ).height

            avg_cost = total_pnl / total_trades if total_trades > 0 else 0.0
            final_eps = float(df["epsilon"].tail(1).item())

            # Steps to reach epsilon floor with tolerance
            eps_floor_rows = df.filter(
                pl.col("epsilon") <= EPSILON_FLOOR_THRESHOLD
            )
            steps_until_eps_05 = (
                int(eps_floor_rows["step"].min())
                if eps_floor_rows.height > 0
                else int(df["step"].max())
            )

            # Timelines for plotting
            trade_timeline = (
                df.filter(pl.col("reward").abs() > TRADE_REWARD_THRESHOLD)[
                    "step"
                ].to_list()
            )

            pnl_curve = list(
                zip(df["step"].to_list(), df["cumulative_pnl"].to_list())
            )

            epsilon_curve = list(
                zip(df["step"].to_list(), df["epsilon"].to_list())
            )

            results[exp_name] = {
                "df": df,
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "avg_cost_per_trade": avg_cost,
                "final_epsilon": final_eps,
                "steps_until_eps_05": steps_until_eps_05,
                "trade_timeline": trade_timeline,
                "pnl_curve": pnl_curve,
                "epsilon_curve": epsilon_curve,
            }
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to compute metrics for {csv_path}: {e}"
            ) from e

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare exploration strategy experiments"
    )
    parser.add_argument(
        "--csv-dir",
        default="output",
        help="Directory containing experiment CSV logs",
    )
    parser.add_argument(
        "--output-dir",
        default="output/exploration_analysis",
        help="Output directory for plots and tables",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results...")
    results = load_experiment_results(csv_dir)
    print(f"Loaded {len(results)} experiments\n")

    # Placeholder for next steps
    print("Analysis functions to be added in next tasks...")


if __name__ == "__main__":
    main()
