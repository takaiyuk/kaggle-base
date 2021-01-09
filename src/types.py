from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd

COMPETITION_NAME = ""
assert COMPETITION_NAME != "", "fill in COMPETITION_NAME, not empty string"


@dataclass
class InputPath:
    prefix: str = f"./input/{COMPETITION_NAME}"
    train: str = f"{prefix}/train.csv"
    test: str = f"{prefix}/test.csv"
    sub: str = f"{prefix}/sample_submission.csv"


@dataclass
class OutputPath:
    prefix: str = "./output"
    confusion: str = f"{prefix}/confusion"
    feature: str = f"{prefix}/feature"
    importance: str = f"{prefix}/importance"
    log: str = f"{prefix}/log"
    model: str = f"{prefix}/model"
    optuna: str = f"{prefix}/optuna"
    submission: str = f"{prefix}/submission"


@dataclass
class FeatureData:
    dfs: List[pd.DataFrame]
    keys: List[str]


@dataclass
class TrainData:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ResultData:
    models: List[Any]
    pred: np.array
