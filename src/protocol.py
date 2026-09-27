"""Fixed research protocols and task definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# Shared


def _tuples(value: Any) -> Any:
    """Keep class orders and task descriptions immutable and hashable."""
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


protocols = _tuples(
    OmegaConf.to_container(
        OmegaConf.load(Path(__file__).resolve().parents[1] / "protocol.yaml"),
        resolve=True,
    )
)


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset: str
    view: str
    classes: tuple[str, ...]
    hf_name: str
    expected_train: int
    expected_val: int
    expected_test: int
    dimensionality: int


TASKS = {
    name: TaskSpec(**values) for name, values in protocols["shared"]["tasks"].items()
}


def resolve_tasks(values: list[str]) -> list[TaskSpec]:
    keys = list(TASKS) if "all" in values else values
    return [TASKS[key] for key in keys]
