import itertools
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import BaseRunConfig
from src.const import ModelPath
from src.data import Loader
from src.utils import Jbl, mkdir, reduce_mem_usage


@dataclass
class TrainData:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FeatureData:
    df: pd.DataFrame
    encoder: Any
    key: List[str]


@dataclass
class ResultData:
    models: List[Any]
    pred: np.array


class Model(metaclass=ABCMeta):
    """https://github.com/upura/ayniy/blob/master/ayniy/model/model.py"""

    def __init__(self, run_cfg: BaseRunConfig, fold_i: int) -> None:
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

    def predict(self, X_te: pd.DataFrame) -> np.ndarray:
        """学習済のモデルでの予測値を返す
        :param X_te: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        if self.model is not None:
            return self.model.predict(X_te, num_iteration=self.model.best_iteration)  # type: ignore
        else:
            raise ValueError("Train model before prediction")

    def predict_proba(self, X_te: pd.DataFrame, is_binary: bool) -> np.ndarray:
        """学習済のモデルでの予測値を返す
        :param X_te: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        if self.model is not None:
            if is_binary:
                return self.model.predict_proba(X_te, num_iteration=self.model.best_iteration)[:, 1]  # type: ignore
            else:
                return self.model.predict_proba(X_te, num_iteration=self.model.best_iteration)  # type: ignore
        else:
            raise ValueError("Train model before prediction")

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
    def __init__(self, run_cfg: BaseRunConfig, logger: Logger):
        self.run_cfg = run_cfg
        self.logger = logger
        self.run_name = run_cfg.basic.name
        self.seed = run_cfg.basic.seed
        self.column = run_cfg.column
        self.feature = run_cfg.feature
        self.kfold = run_cfg.kfold
        self.params = asdict(run_cfg.params)

    def _load(self) -> TrainData:
        loader = Loader()
        if self.fe_cfg.basic.is_debug:
            train = loader.train(nrows=1_000_000)
        else:
            train = loader.train()
        test = loader.test()
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)
        return TrainData(train=train, test=test)

    def _features(self, fold_i: int) -> Dict[str, FeatureData]:
        feature_dict = {}
        feature_list: List[str] = [
            k for k, v in asdict(self.feature).items() if v is True
        ]
        for feature_str in feature_list:
            feature_dict[feature_str] = Jbl.load(
                f"features/{self.run_cfg.kfold.method}-{self.run_cfg.kfold.number}/fold_{fold_i}/{feature_str}.jbl"
            )
        return feature_dict

    def _preprocess(self, df: pd.DataFrame, fold_i: int) -> pd.DataFrame:
        feature_dict = self._features(fold_i)
        key_columns = [
            feature_data.key if type(feature_data.key) == list else [feature_data.key]
            for k, feature_data in feature_dict.items()
        ]
        key_columns = list(itertools.chain.from_iterable(key_columns))
        key_columns = list(np.unique(key_columns))
        selected_columns = key_columns + self.column.external
        X = df.loc[:, selected_columns]
        X = X.fillna(-999)
        for k in feature_dict.keys():
            feature_data: FeatureData = feature_dict[k]
            if k.endswith("Aggregator"):
                X = X.join(feature_data.df, how="left", on=feature_data.key)
            elif k.endswith("Encoder"):
                encoder = feature_data.encoder
                # category-encoder の transform の挙動に注意（reset_index された pd.DataFrame で返ってくる???）
                X[f"{feature_data.key}_encoded"] = encoder.transform(
                    X[feature_data.key].values
                ).values
            else:
                raise ValueError(f"{k} is not supported")
        X = X.drop(key_columns, axis=1)
        return X

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
        plt.savefig(f"{ModelPath.importance}/importance_{self.run_name}.png", dpi=100)

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
        plt.savefig(f"{ModelPath.others}/confusion_matrix_{self.run_name}.png", dpi=100)
