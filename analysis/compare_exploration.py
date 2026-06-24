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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

# Threshold for filtering near-zero rewards to identify actual trades.
# Rewards below this magnitude are treated as no-op/holding actions.
TRADE_REWARD_THRESHOLD = 1e-9

# Epsilon exploration floor value with small tolerance.
# Target epsilon floor is 0.05, but we use 0.051 to account for floating-point
# precision and ensure we capture all rows that have reached the floor.
EPSILON_FLOOR_THRESHOLD = 0.051

# Color map for strategies used across all plotting functions.
# Maps strategy name to consistent color for visual consistency.
STRATEGY_COLORS = {
    "fast_linear": "blue",
    "exponential": "red",
    "logarithmic": "green",
    "episode": "orange",
    "action_local": "purple",
    "parameter_noise": "brown",
}


def _parse_strategy_name(exp_name: str) -> str:
    """Extract canonical strategy name from experiment name.

    Normalizes multi-word strategy names (action_local, parameter_noise)
    by converting their shortened prefixes back to full names.

    Args:
        exp_name: Experiment name like "exp_fast_linear_held"

    Returns:
        Canonical strategy name like "fast_linear" or "action_local"

    Examples:
        exp_fast_linear_held -> fast_linear
        exp_exponential_no_held -> exponential
        exp_action_local_held -> action_local
    """
    # Parse experiment name: exp_<strategy>_<held|no_held>
    parts = exp_name.replace("exp_", "").split("_")

    # Strategy name is everything except last part (held/no_held)
    if len(parts) >= 2 and parts[-1] == "held" and parts[-2] == "no":
        # "no_held" case: exp_fast_linear_no_held -> strategy="fast_linear"
        strategy = "_".join(parts[:-2])
    elif parts[-1] == "held":
        # "held" case: exp_fast_linear_held -> strategy="fast_linear"
        strategy = "_".join(parts[:-1])
    else:
        # Unknown format - return as-is
        strategy = "_".join(parts)

    return strategy


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


def generate_summary_table(results: dict, output_dir: Path) -> None:
    """Generate markdown and CSV summary tables.

    Columns: Strategy, Held?, Total PnL, Trades, Avg Cost, Final ε, Steps to 0.05
    Sorted by: Total PnL (best first)

    Args:
        results: Dict from load_experiment_results()
        output_dir: Directory to save tables

    Raises:
        ValueError: If results is empty or missing required keys
        OSError: If output directory is not writable or file write fails
    """
    # Validate results dict is non-empty
    if not results:
        raise ValueError(
            "Cannot generate summary table: results dict is empty. "
            "Ensure experiment CSVs were loaded successfully."
        )

    # Required keys that must be present in each experiment's metrics
    required_keys = [
        "total_pnl",
        "total_trades",
        "avg_cost_per_trade",
        "final_epsilon",
        "steps_until_eps_05",
    ]

    rows = []
    for exp_name, metrics in results.items():
        # Validate metrics dict contains all required keys
        missing_keys = [key for key in required_keys if key not in metrics]
        if missing_keys:
            raise ValueError(
                f"Experiment '{exp_name}' missing required metrics: {missing_keys}. "
                f"Expected keys: {required_keys}"
            )

        # Parse experiment name: exp_<strategy>_<held|no_held>
        parts = exp_name.replace("exp_", "").split("_")

        # Strategy name is everything except last part (held/no_held)
        if len(parts) >= 2 and parts[-1] == "held" and parts[-2] == "no":
            # "no_held" case: exp_fast_linear_no_held -> strategy="fast_linear", held="No"
            strategy = "_".join(parts[:-2])
            held = "No"
        elif parts[-1] == "held":
            # "held" case: exp_fast_linear_held -> strategy="fast_linear", held="Yes"
            strategy = "_".join(parts[:-1])
            held = "Yes"
        else:
            # Unknown format
            strategy = "_".join(parts)
            held = "Unknown"

        rows.append({
            "Strategy": strategy,
            "Held?": held,
            "Total PnL": f"${metrics['total_pnl']:.2f}",
            "Trades": metrics["total_trades"],
            "Avg Cost": f"${metrics['avg_cost_per_trade']:.3f}",
            "Final ε": f"{metrics['final_epsilon']:.3f}",
            "Steps to 0.05": metrics["steps_until_eps_05"],
        })

    # Sort by PnL (descending - best first)
    rows.sort(
        key=lambda r: float(r["Total PnL"].replace("$", "")),
        reverse=True
    )

    # Write markdown with error handling
    md_path = output_dir / "summary_table.md"
    try:
        with open(md_path, "w") as f:
            # Header
            f.write("# Exploration Strategy Comparison\n\n")
            f.write("| Strategy | Held? | Total PnL | Trades | Avg Cost | Final ε | Steps to 0.05 |\n")
            f.write("|----------|-------|-----------|--------|----------|---------|---------------|\n")

            # Rows
            for row in rows:
                f.write(
                    f"| {row['Strategy']} | {row['Held?']} | {row['Total PnL']} | "
                    f"{row['Trades']} | {row['Avg Cost']} | {row['Final ε']} | "
                    f"{row['Steps to 0.05']} |\n"
                )
    except PermissionError as e:
        raise OSError(
            f"Permission denied writing to {md_path}. "
            f"Check directory permissions for {output_dir}"
        ) from e
    except OSError as e:
        raise OSError(
            f"Failed to write markdown summary to {md_path}: {e}"
        ) from e

    # Write CSV with error handling
    csv_path = output_dir / "summary_table.csv"
    try:
        pl.DataFrame(rows).write_csv(csv_path)
    except PermissionError as e:
        raise OSError(
            f"Permission denied writing to {csv_path}. "
            f"Check directory permissions for {output_dir}"
        ) from e
    except OSError as e:
        raise OSError(
            f"Failed to write CSV summary to {csv_path}: {e}"
        ) from e

    print(f"Summary table saved: {md_path}")
    print(f"Summary CSV saved: {csv_path}")


def plot_pnl_comparison(results: dict, output_dir: Path) -> None:
    """Plot cumulative PnL curves for all experiments.

    X-axis: Training steps
    Y-axis: Cumulative PnL ($)
    Lines: color-coded by strategy, dashed=no_held, solid=held
    """
    # Validate results is non-empty
    if not results:
        raise ValueError(
            "Cannot plot PnL comparison: results dict is empty. "
            "Ensure experiment CSVs were loaded successfully."
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    for exp_name, metrics in results.items():
        # Parse strategy name using helper function
        strategy = _parse_strategy_name(exp_name)
        held = "held" in exp_name and "no_held" not in exp_name

        # Validate pnl_curve is non-empty before unpacking
        if not metrics["pnl_curve"]:
            raise ValueError(
                f"Cannot plot PnL for '{exp_name}': pnl_curve is empty. "
                f"Check that the CSV contains valid step and cumulative_pnl data."
            )

        steps, pnls = zip(*metrics["pnl_curve"])

        ax.plot(
            steps, pnls,
            label=f"{strategy} ({'held' if held else 'no held'})",
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linestyle="-" if held else "--",
            alpha=0.8,
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Exploration Strategy PnL Comparison")
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "pnl_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PnL plot saved: {out_path}")


def plot_epsilon_trajectories(results: dict, output_dir: Path) -> None:
    """Plot epsilon decay curves for all strategies.

    X-axis: Training steps
    Y-axis: Epsilon (exploration rate)
    Lines: one per strategy (averaged across held/no_held variants)
    """
    # Validate results is non-empty
    if not results:
        raise ValueError(
            "Cannot plot epsilon trajectories: results dict is empty. "
            "Ensure experiment CSVs were loaded successfully."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by strategy (average held + no_held)
    by_strategy = {}
    for exp_name, metrics in results.items():
        strategy = _parse_strategy_name(exp_name)

        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(metrics["epsilon_curve"])

    for strategy, curves in by_strategy.items():
        # Validate epsilon_curve is non-empty before unpacking
        if not curves or not curves[0]:
            raise ValueError(
                f"Cannot plot epsilon for strategy '{strategy}': epsilon_curve is empty. "
                f"Check that the CSV contains valid step and epsilon data."
            )

        # Use first curve (they should be identical for same strategy)
        steps, eps_values = zip(*curves[0])
        ax.plot(
            steps, eps_values,
            label=strategy,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=2,
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Epsilon (exploration rate)")
    ax.set_title("Epsilon Decay Trajectories")
    ax.axhline(0.05, color="black", linestyle=":", alpha=0.5, label="Floor (0.05)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "epsilon_decay.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Epsilon plot saved: {out_path}")


def plot_trade_frequency(results: dict, output_dir: Path) -> None:
    """Plot trade frequency over training (rolling window).

    X-axis: Training progress (0-100%)
    Y-axis: Trades per 100 steps (rolling window)
    Shows when each strategy stops exploring and trades decline.
    """
    # Validate results is non-empty
    if not results:
        raise ValueError(
            "Cannot plot trade frequency: results dict is empty. "
            "Ensure experiment CSVs were loaded successfully."
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    window_size = 100  # rolling window for smoothing

    for exp_name, metrics in results.items():
        strategy = _parse_strategy_name(exp_name)
        held = "held" in exp_name and "no_held" not in exp_name

        df = metrics["df"]

        # Validate df is non-empty
        if df.height == 0:
            raise ValueError(
                f"Cannot plot trade frequency for '{exp_name}': DataFrame is empty. "
                f"Check that the CSV contains valid data."
            )
        max_step = df["step"].max()

        # Compute rolling trade frequency
        df = df.with_columns(
            (pl.col("reward").abs() > TRADE_REWARD_THRESHOLD).cast(pl.Int32).alias("is_trade")
        )

        # Group into buckets of window_size steps
        df = df.with_columns(
            (pl.col("step") // window_size).alias("bucket")
        )

        freq = df.group_by("bucket").agg(
            pl.col("is_trade").sum().alias("trades_per_window")
        ).sort("bucket")

        # Convert bucket to progress %
        progress = (freq["bucket"] * window_size / max_step * 100).to_list()
        trades = freq["trades_per_window"].to_list()

        ax.plot(
            progress, trades,
            label=f"{strategy} ({'held' if held else 'no held'})",
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linestyle="-" if held else "--",
            alpha=0.7,
        )

    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel(f"Trades per {window_size} steps")
    ax.set_title("Trade Frequency During Training")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "trade_frequency.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trade frequency plot saved: {out_path}")


def plot_cost_per_trade(results: dict, output_dir: Path) -> None:
    """Bar chart: average cost per trade for each experiment.

    X-axis: Experiment names
    Y-axis: Avg cost per trade ($)
    Sorted by cost (lowest first = most efficient)
    """
    # Validate results is non-empty
    if not results:
        raise ValueError(
            "Cannot plot cost per trade: results dict is empty. "
            "Ensure experiment CSVs were loaded successfully."
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract data
    names = []
    costs = []
    for exp_name, metrics in results.items():
        names.append(exp_name.replace("exp_", "").replace("_", " "))
        costs.append(metrics["avg_cost_per_trade"])

    # Validate we have data to plot
    if not names or not costs:
        raise ValueError(
            "Cannot plot cost per trade: no valid data extracted from results. "
            "Check that experiments contain avg_cost_per_trade metrics."
        )

    # Sort by cost (ascending - best first)
    sorted_pairs = sorted(zip(names, costs), key=lambda x: x[1])
    names, costs = zip(*sorted_pairs)

    # Color bars by cost (red=expensive, orange=medium, green=cheap)
    colors_list = [
        "red" if c < -0.10 else "orange" if c < -0.05 else "green"
        for c in costs
    ]

    ax.barh(names, costs, color=colors_list, alpha=0.7)
    ax.set_xlabel("Avg Cost per Trade ($)")
    ax.set_title("Exploration Cost Efficiency")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(-0.12, color="gray", linestyle=":", alpha=0.5, label="Baseline (-$0.12)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "cost_per_trade.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Cost efficiency plot saved: {out_path}")


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

    print("Generating summary table...")
    generate_summary_table(results, output_dir)

    print("\nGenerating plots...")
    plot_pnl_comparison(results, output_dir)
    plot_epsilon_trajectories(results, output_dir)
    plot_trade_frequency(results, output_dir)
    plot_cost_per_trade(results, output_dir)

    print(f"\nAnalysis complete! Results in {output_dir}/")
    print(f"  - summary_table.md")
    print(f"  - summary_table.csv")
    print(f"  - pnl_comparison.png")
    print(f"  - epsilon_decay.png")
    print(f"  - trade_frequency.png")
    print(f"  - cost_per_trade.png")


if __name__ == "__main__":
    main()
