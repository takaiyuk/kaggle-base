"""
base: -
change: -
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import pandas as pd
import tensorflow as tf
from tensorflow import keras

# https://github.com/tensorflow/tensorflow/issues/45068#issuecomment-732148781
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)


def get_run_name() -> str:
    # __file__ is like '/workspace/src/exp/exp000/config.py'
    try:
        abs_path = __file__
        run_name = abs_path.split("/")[-2]
    except NameError:
        _RUN_NAME = ""
        assert _RUN_NAME != ""
        run_name = _RUN_NAME
    return run_name


RUN_NAME = get_run_name()


@dataclass
class ResultData:
    importance: Optional[pd.DataFrame]
    models: Optional[List[str]]
    scores: Optional[List[float]]
    oof: Optional[pd.DataFrame]
    sub: Optional[pd.DataFrame]
    time: Optional[float]


@dataclass
class Basic:
    run_name: str = f"{RUN_NAME}"
    is_debug: bool = False
    seed: int = 42
    device: str = "cuda" if tf.test.is_gpu_available() else "cpu"


@dataclass(frozen=True)
class Column:
    id: str = ""
    target: str = ""
    sub_target: str = ""


@dataclass
class Feature:
    # num_cols: List[str] = field(default_factory=lambda: [])
    cat_cols: List[str] = field(default_factory=lambda: [])
    drop_cols: List[str] = field(default_factory=lambda: [])
    target_col: str = ""


@dataclass
class Kfold:
    class KfoldMethod(Enum):
        KFOLD = "kf"
        STRATIFIED = "skf"
        GROUP = "gkf"
        STRATIFIED_GROUP = "sgkf"
        TIME_SERIES = "tskf"

    number: int = 5
    method: str = KfoldMethod.STRATIFIED.value
    shuffle: bool = True
    column: str = ""
    seed: int = 42


@dataclass
class LGBMParams:
    class Objective(Enum):
        BINARY = "binary"
        MULTI_CLASS = "multiclass"
        REGRESSION = "regression"
        REGRESSION_L1 = "regression_l1"
        HUBER = "huber"
        MAPE = "mape"

    class Metric(Enum):
        DEFAULT = ""
        NONE = "None"  # Set "None" if feval is used
        MAE = "mae"
        MSE = "mse"
        RMSE = "rmse"
        AUC = "auc"
        BINARY_LOGLOSS = "binary_logloss"
        MULTI_LOGLOSS = "multi_logloss"
        CROSS_ENTROPY = "cross_entropy"

    boosting_type: str = "gbdt"
    tree_learner: str = "serial"
    objective: str = Objective.REGRESSION.value
    num_class: int = 1
    metric: str = Metric.RMSE.value
    n_jobs: int = 8
    n_estimators: int = 100 if Basic.is_debug else 20000
    seed: int = 42
    verbose: int = -1
    learning_rate: float = 0.05
    num_leaves: int = 127
    max_depth: int = -1
    min_child_samples: int = 60
    subsample: float = 0.9
    subsample_freq: int = 3
    colsample_bytree: float = 0.4
    reg_alpha: float = 10.0
    reg_lambda: float = 0.1
    min_data_per_group: int = 100
    max_cat_threshold: int = 32
    cat_l2: float = 10.0
    cat_smooth: float = 10.0
    device: str = "cpu"
    gpu_platform_id: int = -1
    gpu_device_id: int = -1
    early_stopping_rounds: int = 100


@dataclass
class LSTMParams:
    class Verbose(Enum):
        SILENT: int = 0
        PROGRESS_BAR: int = 1

    class CallbackMonitorMode(Enum):
        MIN: str = "min"
        MAX: str = "max"

    @dataclass
    class EarlyStopping:
        monitor: str
        min_delta: float
        patience: int
        mode: str
        verbose: int

    @dataclass
    class Scheduler:
        monitor: str
        factor: float
        patience: int
        mode: str
        min_delta: float
        verbose: int

    def adam():
        return keras.optimizers.Adam(
            learning_rate=0.002,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

    batch_size: int = 1024
    timesteps: int = 60
    units: int = 104
    epochs: int = 100
    verbose: int = Verbose.PROGRESS_BAR.value
    optimizer: Any = adam()
    loss: str = "binary_crossentropy"
    metrics: List[Any] = field(
        default_factory=lambda: [keras.metrics.AUC(num_thresholds=1000, name="auc")]
    )
    early_stopping: EarlyStopping = EarlyStopping(
        monitor="val_auc",
        min_delta=0.0,
        patience=10,
        mode=CallbackMonitorMode.MAX.value,
        verbose=1,
    )
    scheduler: Scheduler = Scheduler(
        monitor="val_auc",
        factor=0.1,
        patience=10,
        mode=CallbackMonitorMode.MAX.value,
        verbose=1,
    )


@dataclass
class RunnerOptions:
    class ModelType(Enum):
        LINEAR: str = "linear"
        GBDT: str = "gbdt"
        MLP: str = "mlp"

    model_type: str = ModelType.GBDT.value


@dataclass
class ModelConfig:
    basic: Basic = Basic()
    column: Column = Column()
    feature: Feature = Feature()
    kfold: Kfold = Kfold()
    params: LGBMParams = LGBMParams()
    options: RunnerOptions = RunnerOptions()
