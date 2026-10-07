"""A complete CPU example; no downloaded data, checkpoints, or accounts needed."""

from pathlib import Path

from befond.training import main


if __name__ == "__main__":
    main(default_config=str(Path(__file__).with_name("toy.json")))
