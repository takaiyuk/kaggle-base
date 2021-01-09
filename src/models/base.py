from abc import ABCMeta, abstractmethod
from dataclasses import asdict
from logging import Logger
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.types import FeatureData, OutputPath, ResultData, TrainData
from src.utils import Jbl, mkdir


class Model(metaclass=ABCMeta):
    """https://github.com/upura/ayniy/blob/master/ayniy/model/model.py"""

    def __init__(self, run_cfg: Any, fold_i: int) -> None:
        self.run_name = run_cfg.basic.name
        self.fold_i = fold_i
        self.categorical_features = run_cfg.column.categorical
        self.params = asdict(run_cfg.params)
        self.model = None

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

    @abstractmethod
    def predict(self, X_te: pd.DataFrame) -> np.ndarray:
        """学習済のモデルでの予測値を返す
        :param X_te: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    def save_model(self, model_path: str) -> None:
        """モデルの保存を行う
        :param path: モデルの保存先パス
        """
        mkdir(model_path)
        Jbl.save(self.model, model_path)

    def load_model(self, model_path: str) -> None:
        """モデルの読み込みを行う
        :param path: モデルの読み込み先パス
        """
        self.model = Jbl.load(model_path)


class AbstractRunner(metaclass=ABCMeta):
    def __init__(self, run_cfg: Any, logger: Logger):
        self.run_cfg = run_cfg
        self.logger = logger
        self.run_name = run_cfg.basic.name
        self.seed = run_cfg.basic.seed
        self.column = run_cfg.column
        self.feature = run_cfg.feature
        self.kfold = run_cfg.kfold
        self.params = asdict(run_cfg.params)

    @abstractmethod
    def _load(self) -> TrainData:
        pass

    @abstractmethod
    def _features(self, fold_i: int) -> Dict[str, FeatureData]:
        pass

    @abstractmethod
    def _preprocess(self, df: pd.DataFrame, fold_i: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def _model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ResultData:
        pass

    @abstractmethod
    def _submit(self, models, y_train, test) -> pd.DataFrame:
        pass

    @abstractmethod
    def run_cv(self):
        pass

    def _feature_importance(
        self, models: List[lgb.Booster], head: Optional[int] = 60
    ) -> None:
        def create_fi(models: List[lgb.Booster]):
            df_fi = pd.DataFrame()
            for i, m in enumerate(models):
                _df = pd.DataFrame()
                _df["fold"] = i
                try:
                    _df["feature"] = m.feature_name_
                    _df["importance"] = m.feature_importances_
                except AttributeError:
                    _df["feature"] = m.feature_name()
                    _df["importance"] = m.feature_importance()
                df_fi = pd.concat([df_fi, _df], axis=0, ignore_index=True)
            return df_fi

        df_fi = create_fi(models)
        order = (
            df_fi.groupby("feature")
            .sum()[["importance"]]
            .sort_values("importance", ascending=False)
            .index[:head]
        )
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.2)))
        sns.boxenplot(
            data=df_fi,
            y="feature",
            x="importance",
            order=order,
            ax=ax,
            palette="viridis",
        )
        fig.tight_layout()
        ax.grid()
        plt.savefig(f"{OutputPath.importance}/importance_{self.run_name}.png", dpi=100)

    def _confusion_matrix(
        self, y_true: np.array, pred_label: np.array, height: float = 0.6, labels=None
    ) -> None:
        conf = confusion_matrix(y_true=y_true, y_pred=pred_label, normalize="true")

        n_labels = len(conf)
        size = n_labels * height
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(conf, cmap="Blues", ax=ax, annot=True, fmt=".2f")
        ax.set_ylabel("Label")
        ax.set_xlabel("Predict")

        if labels is not None:
            ax.set_yticklabels(labels)
            ax.set_xticklabels(labels)
            ax.tick_params("y", labelrotation=0)
            ax.tick_params("x", labelrotation=90)
        fig.tight_layout()
        plt.savefig(
            f"{OutputPath.confusion}/confusion_matrix_{self.run_name}.png", dpi=100
        )
