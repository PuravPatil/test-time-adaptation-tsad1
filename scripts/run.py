"""Run one experiment from one config.

    python scripts/run.py --config configs/experiments/reproduce.yaml
"""
import argparse

from audit.evaluation.harness import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to an experiment YAML")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    run_experiment(args.config, args.config_dir)


if __name__ == "__main__":
    main()
