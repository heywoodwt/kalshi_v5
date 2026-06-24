"""Tests for SLURM script generation.

Verifies that the SLURM generator creates correct experiment scripts
for all exploration strategy combinations.
"""
import tempfile
import shutil
from pathlib import Path
import sys

# Add hpc module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc.generate_exploration_slurm import STRATEGIES, TEMPLATE, main


def test_strategies_count():
    """Verify 6 strategies are defined."""
    assert len(STRATEGIES) == 6
    assert "fast_linear" in STRATEGIES
    assert "exponential" in STRATEGIES
    assert "logarithmic" in STRATEGIES
    assert "episode" in STRATEGIES
    assert "action_local" in STRATEGIES
    assert "parameter_noise" in STRATEGIES


def test_template_contains_placeholders():
    """Verify template has all required placeholders."""
    required_placeholders = [
        "{job_name}",
        "{output_dir}",
        "{experiment_name}",
        "{strategy}",
        "{prioritize_label}",
        "{prioritize_flag}",
        "{run_name}",
    ]
    for placeholder in required_placeholders:
        assert placeholder in TEMPLATE, f"Missing placeholder: {placeholder}"


def test_template_contains_slurm_commands():
    """Verify template has required SLURM directives."""
    assert "#SBATCH" in TEMPLATE
    assert "python -m rl_bot.replay" in TEMPLATE
    assert "--strategy" in TEMPLATE
    assert "--max-loss 500" in TEMPLATE


def test_generates_12_scripts():
    """Verify script generates exactly 12 SLURM files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save original cwd and change to temp directory
        original_cwd = Path.cwd()
        try:
            # Create a minimal hpc directory structure
            hpc_dir = Path(tmpdir) / "hpc"
            hpc_dir.mkdir()

            # Change to temp dir
            import os
            os.chdir(tmpdir)

            # Run generator
            main()

            # Count generated files
            slurm_files = list(hpc_dir.glob("exp_*.slurm"))
            assert len(slurm_files) == 12, f"Expected 12 .slurm files, got {len(slurm_files)}"
        finally:
            os.chdir(original_cwd)


def test_script_naming_convention():
    """Verify generated scripts follow naming pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            hpc_dir = Path(tmpdir) / "hpc"
            hpc_dir.mkdir()

            import os
            os.chdir(tmpdir)
            main()

            slurm_files = list(hpc_dir.glob("exp_*.slurm"))

            # Should have files like exp_fast_linear_held.slurm, exp_fast_linear_no_held.slurm
            # Check that all files end with either _held.slurm or _no_held.slurm
            for f in slurm_files:
                assert f.name.endswith("_held.slurm") or f.name.endswith("_no_held.slurm"), \
                    f"File {f.name} doesn't match expected naming pattern"

            # Each strategy should appear twice (held and no_held)
            for strategy in STRATEGIES:
                matching = [f for f in slurm_files if strategy in f.name]
                assert len(matching) == 2, f"Strategy {strategy} should have 2 variants"
        finally:
            os.chdir(original_cwd)


def test_script_content_has_strategy():
    """Verify generated script contains correct strategy flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            hpc_dir = Path(tmpdir) / "hpc"
            hpc_dir.mkdir()

            import os
            os.chdir(tmpdir)
            main()

            # Check one specific script
            fast_linear_script = hpc_dir / "exp_fast_linear_held.slurm"
            assert fast_linear_script.exists()

            content = fast_linear_script.read_text()
            assert "--strategy fast_linear" in content
            assert "--prioritize-held" in content
            assert "exp_fast_linear_held" in content
        finally:
            os.chdir(original_cwd)


def test_script_content_no_held_variant():
    """Verify no_held variant doesn't have --prioritize-held flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            hpc_dir = Path(tmpdir) / "hpc"
            hpc_dir.mkdir()

            import os
            os.chdir(tmpdir)
            main()

            # Check no_held variant
            no_held_script = hpc_dir / "exp_exponential_no_held.slurm"
            assert no_held_script.exists()

            content = no_held_script.read_text()
            assert "--strategy exponential" in content
            # Should not have --prioritize-held flag on the command line
            lines = content.split("\n")
            for line in lines:
                if "python -m rl_bot.replay" in line:
                    # Find the full command (may span multiple lines)
                    idx = lines.index(line)
                    command = line
                    while idx < len(lines) - 1 and "\\" in lines[idx]:
                        idx += 1
                        command += " " + lines[idx]
                    assert "--prioritize-held" not in command
        finally:
            os.chdir(original_cwd)


def test_output_directory_created():
    """Verify exploration_exps output directory is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            hpc_dir = Path(tmpdir) / "hpc"
            hpc_dir.mkdir()

            import os
            os.chdir(tmpdir)
            main()

            output_dir = Path("hpc/exploration_exps")
            assert output_dir.exists()
            assert output_dir.is_dir()
        finally:
            os.chdir(original_cwd)


def test_script_is_executable():
    """Verify generated scripts have executable permissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            hpc_dir = Path(tmpdir) / "hpc"
            hpc_dir.mkdir()

            import os
            os.chdir(tmpdir)
            main()

            slurm_files = list(hpc_dir.glob("exp_*.slurm"))
            for script in slurm_files:
                # Check executable bit
                assert os.access(script, os.X_OK), f"{script.name} is not executable"
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Run tests manually
    test_strategies_count()
    test_template_contains_placeholders()
    test_template_contains_slurm_commands()
    test_generates_12_scripts()
    test_script_naming_convention()
    test_script_content_has_strategy()
    test_script_content_no_held_variant()
    test_output_directory_created()
    test_script_is_executable()
    print("All tests passed!")
