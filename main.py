"""Run any experiment with one configuration-driven entry point."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="acdc_2d")
def main(config: DictConfig) -> None:
    # Keep --help and --cfg usable without importing the training stack.
    from src.train import run

    run(config)


if __name__ == "__main__":
    main()
