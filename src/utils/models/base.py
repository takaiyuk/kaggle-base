import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.file import mkdir
from src.utils.joblib import Jbl


class BaseModel(metaclass=ABCMeta):
    """https://github.com/upura/ayniy/blob/master/ayniy/model/model.py"""

    def __init__(
        self,
        params: Dict[str, Any],
        categorical_features: Optional[List[str]] = None,
    ) -> None:
        self.model: Any = None
        self.history: Any = None
        self.params = params
        self.categorical_features = categorical_features
        self._validate_categorical_features()

    @abstractmethod
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        :param X_tr: 学習データの特徴量
        :param y_tr: 学習データの目的変数
        :param X_val: バリデーションデータの特徴量
        :param y_val: バリデーションデータの目的変数
        """
        pass

    def predict(self, X_te: pd.DataFrame) -> np.ndarray:
        """学習済のモデルでの予測値を返す
        :param X_te: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        if self.model is None:
            raise ValueError("train model before predict")
        return self.model.predict(X_te)

    def save_model(self, model_path: str) -> None:
        """モデルの保存を行う
        :param path: モデルの保存先パス
        """
        model_path_dir = os.path.dirname(model_path)
        mkdir(model_path_dir)
        Jbl.save(self.model, model_path)

    def load_model(self, model_path: str) -> None:
        """モデルの読み込みを行う
        :param path: モデルの読み込み先パス
        """
        self.model = Jbl.load(model_path)

    def save_params(self, path: str) -> None:
        """path: str = f'{OutputPath.optuna}/best_params.json'"""
        with open(path, "w") as f:
            json.dump(self.model.params, f, indent=4, separators=(",", ": "))

    def _validate_categorical_features(self):
        class_name = self.__class__.__name__
        categorical_handle_models = ["LGBMModel", "LGBMOptunaModel"]
        if (
            self.categorical_features is not None
            and class_name not in categorical_handle_models
        ):
            print(f"[WARNING]{class_name} cannot handle categorical features")
