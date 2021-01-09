from dataclasses import dataclass, field
from typing import List

from src.config import BaseRunConfig


@dataclass
class Basic:
    description: str = "Lightgbm Classifier"
    name: str = "run000"
    is_debug: bool = False
    seed: int = 42


@dataclass
class Column:
    categorical: List[str] = field(default_factory=lambda: [""])
    target: str = ""
    numerical: List[str] = field(default_factory=lambda: [""])


@dataclass
class Feature:
    features: bool = True


@dataclass
class Kfold:
    number: int = 2
    method: str = "tskf"
    shuffle: bool = True
    columns: List[str] = field(default_factory=lambda: [""])


@dataclass
class Model:
    eval_metric: str = ""
    name: str = "ModelLGBM"


@dataclass
class LGBMParams:
    boosting_type: str = "gbdt"
    tree_learner: str = "serial"
    objective: str = "binary"
    metric: str = "binary_logloss"
    n_jobs: int = 8
    seed: int = 42
    num_boost_round: int = 10000
    early_stopping_rounds: int = 100
    verbose: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 47
    max_depth: int = 6
    min_child_samples: int = 20
    subsample: float = 0.9
    subsample_freq: int = 3
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.01
    reg_lambda: float = 0.1
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    cat_l2: float = 10.0
    cat_smooth: float = 10.0
    device: str = "cpu"
    gpu_platform_id: int = -1
    gpu_device_id: int = -1


@dataclass
class RunConfig(BaseRunConfig):
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    kfold: Kfold = Kfold()
    model: Model = Model()
    params: LGBMParams = LGBMParams()
