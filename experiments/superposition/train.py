"""Train BeFOND with the current superposition defaults."""

from befond.training import main


if __name__ == "__main__":
    main(default_config="configs/superposition.json")
