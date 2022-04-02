import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from keras import callbacks

from src.utils.file import mkdir
from src.utils.models.base import BaseModel


@dataclass
class EarlyStoppingOptions:
    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 10
    mode: str = "min"
    verbose: int = 0


@dataclass
class SchedulerOptions:
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 10
    mode: str = "min"
    verbose: int = 0


@dataclass
class LSTMOptions:
    batch_size: int
    timesteps: int
    units: int
    epochs: int
    verbose: int
    optimizer: str
    loss: str
    metrics: List[Any]
    early_stopping: EarlyStoppingOptions = EarlyStoppingOptions()
    scheduler: SchedulerOptions = SchedulerOptions()


class BaseMLPModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        options = self._build_options(self.params)
        self.model = self._build_model(options)
        self.history = self.model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            batch_size=options.batch_size,
            epochs=options.epochs,
            verbose=options.verbose,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor=options.early_stopping.monitor,
                    min_delta=options.early_stopping.min_delta,
                    patience=options.early_stopping.patience,
                    mode=options.early_stopping.mode,
                    verbose=options.early_stopping.verbose,
                    restore_best_weights=True,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor=options.scheduler.monitor,
                    factor=options.scheduler.factor,
                    patience=options.scheduler.patience,
                    mode=options.scheduler.mode,
                    verbose=options.scheduler.verbose,
                ),
            ],
        )

    @abstractmethod
    def _build_options(self, params: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def _build_model(self, options: Any) -> Any:
        pass

    def save_model(self, model_path: str) -> None:
        """モデルの重みの保存を行う
        :param path: モデルの重みの保存先パス
        """
        model_path_dir = os.path.dirname(model_path)
        mkdir(model_path_dir)
        self._save_model_weights(model_path)

    def load_model(self, model_path: str) -> None:
        """モデルの重みの読み込みを行う
        :param path: モデルの重みの読み込み先パス
        """
        options = self._build_options(self.params)
        self.model = self._build_model(options)
        self._load_model_weights(model_path)

    def _save_model_weights(self, checkpoint_path: str) -> None:
        self.model.save_weights(checkpoint_path)

    def _load_model_weights(self, checkpoint_path: str) -> None:
        self.model.load_weights(checkpoint_path)
