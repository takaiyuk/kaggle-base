from dataclasses import dataclass, field
from typing import List

from src.config import BaseFeConfig


@dataclass
class Basic:
    name: str = "fe000"
    is_debug: bool = False
    seed: int = 42


@dataclass
class Column:
    categorical: List[str] = field(default_factory=lambda: [""])
    target: str = ""


@dataclass
class Feature:
    features: bool = True


@dataclass
class Kfold:
    number: int = 2
    method: str = "tskf"
    column: str = ""


@dataclass
class FeConfig(BaseFeConfig):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    kfold: Kfold = Kfold()
