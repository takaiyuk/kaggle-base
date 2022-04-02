import datetime
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Generator, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from tensorflow import keras
from tensorflow.keras import layers

from src.exp.common.const import InputPath, OutputPath
from src.exp.exp000.config import ModelConfig, ResultData
from src.utils.joblib import Jbl
from src.utils.kfold import StratifiedGroupKFold
from src.utils.logger import DefaultLogger, Logger
from src.utils.models.gbdt import LGBMModel
from src.utils.models.mlp import (
    BaseMLPModel,
    EarlyStoppingOptions,
    LSTMOptions,
    SchedulerOptions,
)
from src.utils.notify import LINENotify
from src.utils.seed import fix_seed
from src.utils.time import timer

sns.set_style("whitegrid")


def validate_config(model_config: ModelConfig) -> None:
    def _validate_run_name(model_config: ModelConfig) -> None:
        # If you want to remove models, run `rm output/models/*expXXX*` in the root dir.
        past_sessions = [
            x.split("_")[0]
            for x in os.listdir(OutputPath.models)
            if x.endswith("_0.jbl") or x.endswith("_0.jbl.index")
        ]
        assert (
            model_config.basic.run_name not in past_sessions
        ), f"run `rm output/models/*{model_config.basic.run_name}*` if you want to re-run"

    def _validate_device(model_config: ModelConfig) -> None:
        # assert model_config.basic.device == "cuda"
        pass

    _validate_run_name(model_config)
    _validate_device(model_config)


def load_csv(path: str, verbose: bool = False, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    if verbose:
        print(f"{path.split('/')[-1]}\tshape: {df.shape}")
    return df


class FeatureGenerator:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def generate(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        raise NotImplementedError
        return df


class KFoldGenerator:
    def __init__(self, cfg: ModelConfig) -> None:
        self.kfold = cfg.kfold

    def _get_instance(self) -> Generator:
        if self.kfold.method == "kf":
            kf = model_selection.KFold(
                n_splits=self.kfold.number,
                shuffle=self.kfold.shuffle,
                random_state=self.kfold.seed,
            )
        elif self.kfold.method == "skf":
            kf = model_selection.StratifiedKFold(
                n_splits=self.kfold.number,
                shuffle=self.kfold.shuffle,
                random_state=self.kfold.seed,
            )
        elif self.kfold.method == "gkf":
            kf = model_selection.GroupKFold(n_splits=self.kfold.number)
        elif self.kfold.method == "sgkf":
            kf = StratifiedGroupKFold(
                n_splits=self.kfold.number,
                random_state=self.kfold.seed,
            )
        elif self.kfold.method == "tskf":
            kf = model_selection.TimeSeriesSplit(n_splits=self.kfold.number)
        else:
            raise ValueError(f"{self.kfold.method} is not supported")
        return kf

    def generate(self, df_train: pd.DataFrame) -> pd.DataFrame:
        kf = self._get_instance()
        kf_generator = kf.split(df_train, df_train[self.kfold.column])
        for fold_i, (_, val_idx) in enumerate(kf_generator):
            df_train.loc[val_idx, "fold"] = fold_i
        if self.kfold.method == "tskf":
            df_train = df_train.fillna({"fold": 0})
        df_train = df_train.assign(fold=df_train["fold"].astype(int))
        return df_train


class LSTMModel(BaseMLPModel):
    def _build_options(self, params: Dict[str, Any]) -> LSTMOptions:
        return LSTMOptions(
            batch_size=params["batch_size"],
            timesteps=params["timesteps"],
            units=params["units"],
            epochs=params["epochs"],
            verbose=params["verbose"],
            optimizer=params["optimizer"],
            loss=params["loss"],
            metrics=params["metrics"],
            early_stopping=EarlyStoppingOptions(
                monitor=params["early_stopping"]["monitor"],
                min_delta=params["early_stopping"]["min_delta"],
                patience=params["early_stopping"]["patience"],
                mode=params["early_stopping"]["mode"],
                verbose=params["early_stopping"]["verbose"],
            ),
            scheduler=SchedulerOptions(
                monitor=params["scheduler"]["monitor"],
                factor=params["scheduler"]["factor"],
                patience=params["scheduler"]["patience"],
                mode=params["scheduler"]["mode"],
                verbose=params["scheduler"]["verbose"],
            ),
        )

    def _build_model(self, options: LSTMOptions) -> Any:
        # https://www.kaggle.com/code/ted0071/tps-apr-2022-keras-lstm-model#LSTM-Model
        model = keras.Sequential(
            [
                layers.Input(
                    shape=(options.timesteps, options.units), name="input_layer"
                ),
                layers.BatchNormalization(),
                layers.Bidirectional(layers.LSTM(options.units, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
                layers.Conv1D(64, 3),
                layers.MaxPooling1D(),
                layers.Conv1D(128, 3),
                layers.GlobalMaxPooling1D(),
                layers.Dense(128, activation="swish"),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="lstm_model",
        )
        model.compile(
            optimizer=options.optimizer,
            loss=options.loss,
            metrics=options.metrics,
        )
        return model


class BaseRunner:
    def __init__(self, cfg: ModelConfig, logger: Optional[Logger] = None):
        self.cfg = cfg
        self.params = cfg.params
        self.options = cfg.options
        if logger is not None:
            self.logger = logger
        else:
            self.logger = DefaultLogger()
        self.logger.info(self.cfg)

    def _evaluate(
        self, y_true: np.array, y_pred: np.array, verbose: bool = False
    ) -> float:
        def evaluete(t: np.array, p: np.array) -> float:
            raise NotImplementedError

        score = evaluete(y_true, y_pred)
        if verbose:
            self.logger.info(f"Score: {score:<.5f}")
        return score

    def run_cv(self, df: pd.DataFrame) -> None:
        raise NotImplementedError


class TrainRunner(BaseRunner):
    def run_cv(self, train: pd.DataFrame) -> None:
        self.logger.info(f"Runner: {self.__class__.__name__}")
        self.logger.info(f"debug mode: {self.cfg.basic.is_debug}")
        self.logger.info(f"start time: {datetime.datetime.now()}")
        oof_df = pd.DataFrame()
        scores: List[float] = []
        for n_fold in range(self.cfg.kfold.number):
            start_time = time.time()
            _oof_df = self._train(train, n_fold)
            oof_df = pd.concat([oof_df, _oof_df])
            elapsed = time.time() - start_time
            score = self._evaluate(
                _oof_df[self.cfg.feature.target_col], _oof_df["pred"], verbose=True
            )
            scores.append(score)
            self.logger.info(f"========== fold: {n_fold} result ==========")
            self.logger.info(f"fold{n_fold} time: {elapsed/60:.0f}min.")
            if hasattr(self.logger, "result"):
                self.logger.result(f"Fold {n_fold} Score: {score:<.5f}")
        score = self._evaluate(
            oof_df[self.cfg.feature.target_col], oof_df["pred"], verbose=True
        )
        result_data = ResultData(
            importance=None,
            models=[
                f"{OutputPath.models}/{self.cfg.basic.run_name}_{n_fold}.jbl"
                for i in range(self.cfg.kfold.number)
            ],
            scores=scores,
            oof=oof_df,
            sub=None,
            time=None,
        )
        Jbl.save(
            result_data,
            f"{OutputPath.models}/result_data_{self.cfg.basic.run_name}.jbl",
        )
        self.logger.info("========== CV ==========")
        if hasattr(self.logger, "result"):
            self.logger.result(f"CV Score: {score:<.5f}")

    def _train(self, train: pd.DataFrame, n_fold: int) -> pd.DataFrame:
        self.logger.info(f"fold: {n_fold}")

        trn_idx = train[train["fold"] != n_fold].index.tolist()
        val_idx = train[train["fold"] == n_fold].index.tolist()
        train_fold = train.iloc[trn_idx]
        valid_fold = train.iloc[val_idx]
        X_tr = train_fold.drop(
            self.cfg.feature.drop_cols + [self.cfg.feature.target_col, "fold"], axis=1
        )
        y_tr = train_fold[self.cfg.feature.target_col]
        X_val = valid_fold.drop(
            self.cfg.feature.drop_cols + [self.cfg.feature.target_col, "fold"], axis=1
        )
        y_val = valid_fold[self.cfg.feature.target_col]

        model = LGBMModel(
            params=asdict(self.params),
            categorical_features=self.cfg.feature.cat_cols,
        )
        with timer(f"fold{n_fold}", self.logger):
            model.train(
                X_tr,
                y_tr,
                X_val,
                y_val,
                # fobj=self._fobj,
                # feval=self._feval,
            )
        pred_valid = model.predict(X_val)
        valid_fold = valid_fold[[self.cfg.feature.target_col]].assign(pred=pred_valid)
        model.save_model(f"{OutputPath.models}/{self.cfg.basic.run_name}_{n_fold}.jbl")
        return valid_fold

    def _fobj(preds, dtrain):
        # QWK: https://zenn.dev/jackthekaggler/articles/cf988ca341e34ed83034
        raise NotImplementedError
        num_classes = 4
        a = 1.85
        b = 1.63
        labels = dtrain.get_label()
        preds = preds.clip(0, num_classes - 1)
        f = 1 / 2 * np.sum((preds - labels) ** 2)
        g = 1 / 2 * np.sum((preds - a) ** 2 + b)
        df = preds - labels
        dg = preds - a
        grad = (df / g - f / dg * g ** 2) * len(labels)
        hess = np.ones(len(labels))
        return grad, hess

    def _feval(self, preds: np.array, data: lgb.Dataset):
        raise NotImplementedError
        name = "NAME"
        y_true = data.get_label()
        score = self._evaluate(y_true, preds, verbose=False)
        is_higher_better = False
        # name, result, is_higher_better
        return name, score, is_higher_better


class InferenceRunner(BaseRunner):
    def run_cv(self, test: pd.DataFrame) -> None:
        self.logger.info(f"Runner: {self.__class__.__name__}")
        preds: List[np.array] = []
        df_fi: List[pd.DataFrame] = []
        for n_fold in range(self.cfg.kfold.number):
            preds_fold, _df_fi = self._test(test, n_fold)
            preds.append(preds_fold)
            if self.options.model_type == "gbdt":
                df_fi.append(_df_fi)
        Jbl.save(preds, f"{OutputPath.models}/preds_test_{self.cfg.basic.run_name}.jbl")
        if self.options.model_type == "gbdt":
            Jbl.save(
                df_fi, f"{OutputPath.importances}/df_fi_{self.cfg.basic.run_name}.jbl"
            )
        preds_mean = np.mean(preds, axis=0)
        assert len(preds_mean) == len(test)

        self._submit(preds_mean)
        if self.options.model_type == "gbdt":
            self._plot_feature_importance(
                df_fi,
                filepath=f"{OutputPath.importances}/{self.cfg.basic.run_name}.png",
                head=60,
            )
        self._visualize_prediction()

    def _test(
        self, test: pd.DataFrame, n_fold: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info(f"fold: {n_fold}")
        X_te = test.drop(self.cfg.feature.drop_cols, axis=1)
        model = LGBMModel(
            params=asdict(self.params),
            categorical_features=self.cfg.feature.cat_cols,
        )
        model.load_model(f"{OutputPath.models}/{self.cfg.basic.run_name}_{n_fold}.jbl")
        pred_test = model.predict(X_te)
        if self.options.model_type == "gbdt":
            df_fi = model.feature_importance()
        else:
            df_fi = None
        return pred_test, df_fi

    def _submit(self, preds: np.array) -> None:
        df_sub = load_csv(InputPath.sample_submission)
        df_sub.loc[:, self.cfg.column.sub_target] = preds
        self.logger.info(df_sub.head())
        df_sub = df_sub.astype(
            {self.cfg.column.id: int, self.cfg.column.sub_target: float}
        )
        path = f"{OutputPath.submissions}/submission_{self.cfg.basic.run_name}.csv"
        df_sub.to_csv(path, index=False)
        self.logger.info("submission.csv created")

    def _plot_feature_importance(
        self, df_fi: pd.DataFrame, filepath: str, head: int = 60
    ) -> None:
        df_fi = pd.concat(df_fi, axis=0, ignore_index=True)
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
        plt.savefig(filepath, dpi=100)
        plt.close()

    def _visualize_prediction(self) -> None:
        result_data = Jbl.load(
            f"{OutputPath.models}/result_data_{self.cfg.basic.run_name}.jbl"
        )
        oof = result_data.oof
        sub = load_csv(
            f"{OutputPath.submissions}/submission_{self.cfg.basic.run_name}.csv"
        )
        self.logger.info(oof.head())
        self.logger.info(sub.head())
        plt.figure(figsize=(8, 6))
        sns.distplot(oof[self.cfg.column.target], label="oof_target")
        sns.distplot(oof[self.cfg.column.sub_target], label="oof_pred")
        sns.distplot(sub[self.cfg.column.target], label="sub")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{OutputPath.models}/pred_hist_{self.cfg.basic.run_name}.png", dpi=100
        )
        plt.close()


def run(is_debug: bool):
    fix_seed()
    model_config = ModelConfig()
    validate_config(model_config)
    model_config.basic.is_debug = is_debug
    run_name = model_config.basic.run_name
    if model_config.basic.is_debug:
        logger = DefaultLogger()
    else:
        logger = Logger(
            f"{OutputPath.logs}/{run_name}/general.log",
            f"{OutputPath.logs}/{run_name}/result.log",
            run_name,
        )
    logger.info(f"debug mode: {model_config.basic.is_debug}")

    df_train = load_csv(InputPath.train)
    if model_config.basic.is_debug:
        df_train = df_train[df_train[model_config.column.id] < 1000]
    with timer("Process train feature engineering", logger):
        fg = FeatureGenerator(model_config)
        train = fg.generate(df_train, is_train=True)
    with timer("Process kfold split", logger):
        kg = KFoldGenerator(model_config)
        train = kg.generate(train)
    TrainRunner(model_config, logger).run_cv(train)

    test = load_csv(InputPath.test)
    with timer("Process test feature engineering", logger):
        test = fg.generate(test, is_train=False)
    InferenceRunner(model_config, logger).run_cv(test)

    if not model_config.basic.is_debug:
        with open(f"{OutputPath.logs}/{run_name}/result.log", "r") as f:
            message = f.read()
            LINENotify().send_message(message)


def main(is_debug: bool):
    run(is_debug)
