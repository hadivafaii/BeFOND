"""Train BeFOND with the current gemma defaults."""

from befond.training import main


if __name__ == "__main__":
    main(default_config="configs/gemma.json")
