"""CLI entry: python -m sakigo.train --config run.toml [overrides]."""

from sakigo.train.config import parse_args
from sakigo.train.trainer import train_from_config


def main() -> None:
    config = parse_args()
    final = train_from_config(config)
    print(f"final checkpoint: {final}")


if __name__ == "__main__":
    main()
