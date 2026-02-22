from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    DEV = "dev"
    PROD = "prod"


@dataclass
class InspectorConfig:
    mode: Mode = Mode.DEV
    sample_rate: float = 1.0
    store_path: str | None = None
    max_records: int = 1000
