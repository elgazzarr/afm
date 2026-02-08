"""Command-line interface for AFM experiments."""

import argparse
import warnings
import sys
from pathlib import Path

from .experiments.runner import ExperimentRunner

# Suppress JAX warnings
warnings.filterwarnings("ignore", message="Unable to parse backend config for custom call")


def _resolve_config(config_ref: str) -> Path:
    """Resolve a config reference to a file path.

    Accepts either:
      - A direct file path (absolute or relative)
      - A bundled config name (e.g. 'afm_vs_fm_single')
    """
    path = Path(config_ref)
    if path.exists():
        return path

    # Try as bundled config name
    try:
        import importlib.resources as pkg_resources
        configs_dir = pkg_resources.files("afm") / "configs" / "experiments"
        # Try exact name, then with .yaml suffix
        for candidate in [config_ref, f"{config_ref}.yaml"]:
            cfg_path = configs_dir / candidate
            if cfg_path.is_file():
                return Path(str(cfg_path))
    except Exception:
        pass

    # Fallback: look relative to this package
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir / "configs" / "experiments" / config_ref,
        pkg_dir / "configs" / "experiments" / f"{config_ref}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Config not found: '{config_ref}'. "
        f"Pass a file path or a bundled config name (see 'afm list')."
    )


def _list_bundled_configs():
    """List all bundled experiment configs."""
    pkg_dir = Path(__file__).parent
    configs_dir = pkg_dir / "configs" / "experiments"
    if configs_dir.exists():
        configs = sorted(configs_dir.glob("*.yaml"))
        return [c.stem for c in configs]
    return []


def cmd_run(args):
    """Run an experiment."""
    config_path = _resolve_config(args.config)

    try:
        runner = ExperimentRunner(str(config_path), args.seed)

        # Apply overrides
        if args.system and runner.experiment_type == 'afm_vs_fm_single_system':
            runner.config.system_name = args.system
            print(f"Override: system = {args.system}")

        if args.num_epochs:
            runner.config.training_num_epochs = args.num_epochs
            print(f"Override: num_epochs = {args.num_epochs}")

        if args.nfe:
            runner.config.flow_nfe = args.nfe
            print(f"Override: nfe = {args.nfe}")

        # Recompute derived fields after overrides
        if any([args.system, args.num_epochs, args.nfe]):
            runner.config._compute_derived_fields()

        results = runner.run()

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Experiment: {runner.config.experiment_name}")
        print(f"Type: {runner.experiment_type}")
        print(f"Results saved to: {getattr(runner.config, 'experiment_results_dir', 'results/')}")

        if isinstance(results, dict):
            if 'AFM' in results and 'FM' in results:
                afm_crps = results['AFM'].get('crps_prediction', 'N/A')
                fm_crps = results['FM'].get('crps_prediction', 'N/A')
                print(f"\n  AFM CRPS: {afm_crps}")
                print(f"  FM CRPS:  {fm_crps}")
                if afm_crps != 'N/A' and fm_crps != 'N/A':
                    winner = "AFM" if afm_crps < fm_crps else "FM"
                    print(f"  Winner: {winner}")

        print("=" * 80)

    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_list(args):
    """List available experiments and configs."""
    print("Available Experiment Types:")
    print("=" * 50)
    experiments = ExperimentRunner.list_available_experiments()
    for exp_type, description in experiments.items():
        print(f"  {exp_type:<30} {description}")

    print("\nBundled Configs:")
    print("-" * 50)
    for name in _list_bundled_configs():
        print(f"  {name}")

    print(f"\nUsage: afm run <config_name_or_path>")


def cmd_validate(args):
    """Validate a config file."""
    config_path = _resolve_config(args.config)

    try:
        runner = ExperimentRunner(str(config_path), 0)
        is_valid = runner.validate_config()
        if is_valid:
            print(f"Configuration is valid: {config_path}")
            print(f"  Experiment Type: {runner.experiment_type}")
            print(f"  Experiment Name: {runner.config.experiment_name}")
        else:
            print(f"Configuration is invalid: {config_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="afm",
        description="AFM - Autoregressive Flow Matching for dynamical systems",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "config",
        help="Config file path or bundled config name (e.g. afm_vs_fm_single)",
    )
    run_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    run_parser.add_argument(
        "--system",
        choices=[
            "lorenz", "lotka_volterra", "vanderpol", "harmonic_oscillator",
            "brusselator", "fitzhugh_nagumo", "duffing",
        ],
        help="Override system name",
    )
    run_parser.add_argument("--num-epochs", type=int, help="Override training epochs")
    run_parser.add_argument("--nfe", type=int, help="Override NFE")
    run_parser.set_defaults(func=cmd_run)

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List experiments and configs")
    list_parser.set_defaults(func=cmd_list)

    # --- validate ---
    val_parser = subparsers.add_parser("validate", help="Validate a config file")
    val_parser.add_argument("config", help="Config file path or bundled config name")
    val_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)
