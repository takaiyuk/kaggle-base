import warnings
from typing import Any, List

import numpy as np
import pandas as pd

from src.const import DataPath, ModelPath
from src.models import (
    AbstractRunner,
    CatClassifierModel,
    CatRegressorModel,
    LGBMModel,
    LGBMOptunaModel,
    LRModel,
    ResultData,
    XGBModel,
)
from src.models.evaluate import AUC, MAE, PRAUC, RMSE
from src.models.kfold import generate_kf
from src.utils import Jbl, check_exist

warnings.filterwarnings("ignore")


class TrainRunner(AbstractRunner):
    def _model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ResultData:
        models = []
        oof_pred = np.zeros((len(y_train)))
        kf = generate_kf(self.run_cfg)
        if self.kfold.method.endswith("gkf"):
            assert (
                self.kfold.columns != ""
            ), f"{self.kfold.columns} should not be empty list when group kfold"
            kf_generator = kf.split(
                X_train, y_train, groups=X_train[self.kfold.columns],
            )
        else:
            kf_generator = kf.split(X_train, y_train)
        for fold_i, (tr_idx, val_idx) in enumerate(kf_generator):
            print("=" * 60)
            print(f"fold: {fold_i}")
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            X_tr = self._preprocess(X_tr, fold_i)
            X_val = self._preprocess(X_val, fold_i)
            if self.run_cfg.model.name == "ModelLGBM":
                model = LGBMModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelLGBMOptuna":
                model = LGBMOptunaModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                self.logger.info(f"Best params by optuna: {model.model.params}")
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelXGB":
                model = XGBModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelCBClf":
                model = CatClassifierModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict_proba(X_val, is_binary=True)
            elif self.run_cfg.model.name == "ModelCBReg":
                model = CatRegressorModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            else:
                raise ValueError(f"{self.run_cfg.model.name} is not supported")
            if self.run_cfg.model.name == "ModelLGBMOptuna":
                oof_pred[val_idx] = y_pred
            else:
                oof_pred[val_idx] = y_pred[:, 1]
            models.append(model.model)
        if self.run_cfg.model.eval_metric == "auc":
            score = AUC(y_train, oof_pred)
        elif self.run_cfg.model.eval_metric == "mae":
            score = MAE(y_train, oof_pred)
        elif self.run_cfg.model.eval_metric == "pruac":
            score = PRAUC(y_train, oof_pred)
        elif self.run_cfg.model.eval_metric == "rmse":
            score = RMSE(y_train, oof_pred)
        else:
            raise ValueError(f"{self.model.eval_metric} is not supported")
        self.logger.info(f"cv score: {score:.6f}")
        results = ResultData(models=models, pred=oof_pred)
        Jbl.save(results, f"{ModelPath.model}/results_{self.run_name}.jbl")
        return results

    def _submit(
        self, models: List[Any], y_train: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        pred_test = np.zeros((len(test)))
        for fold_i in range(self.kfold.number):
            X_test = self._preprocess(test, fold_i)
            if self.run_cfg.model.name == "ModelLGBM":
                y_pred = models[fold_i].predict(X_test.values)
            elif self.run_cfg.model.name == "ModelLGBMOptuna":
                y_pred = models[fold_i].predict(X_test.values)
            elif self.run_cfg.model.name == "ModelXGB":
                y_pred = models[fold_i].predict(X_test.values)
            elif self.run_cfg.model.name == "ModelCBClf":
                y_pred = models[fold_i].predict_proba(X_test.values, is_binary=True)
            elif self.run_cfg.model.name == "ModelCBReg":
                y_pred = models[fold_i].predict(X_test.values)
            else:
                raise ValueError(f"{self.run_cfg.model.name} is not supported")
            pred_test += y_pred / self.kfold.number
        self.logger.info(pred_test)
        sub = pd.DataFrame(pred_test, columns=[self.run_cfg.column.target])
        results = ResultData(models=None, pred=pred_test)
        Jbl.save(results, f"{ModelPath.model}/results_test_{self.run_name}.jbl")
        sub.to_csv(
            f"{ModelPath.submission}/submission_{self.run_name}.csv", index=False
        )

    def run_cv(self):
        suffix = "_debug" if self.run_cfg.basic.is_debug else ""
        if check_exist(f"{DataPath.interim.train}{suffix}.jbl") and check_exist(
            f"{DataPath.interim.test}{suffix}.jbl"
        ):
            train = Jbl.load(f"{DataPath.interim.train}{suffix}.jbl")
            test = Jbl.load(f"{DataPath.interim.test}{suffix}.jbl")
        else:
            train_data = self._load()
            train = train_data.train
            test = train_data.test
            del train_data
            Jbl.save(train, f"{DataPath.interim.train}{suffix}.jbl")
            Jbl.save(test, f"{DataPath.interim.test}{suffix}.jbl")
        X_train = train.drop(self.run_cfg.column.target, axis=1)
        y_train = train[self.run_cfg.column.target]

        results = self._model(X_train, y_train)
        if self.run_cfg.model.name == "ModelLGBM":
            self._feature_importance(results.models)
        self._submit(results.models, y_train, test)


class StackingRunner(AbstractRunner):
    def _model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ResultData:
        models = []
        oof_pred = np.zeros((len(y_train)))
        kf = generate_kf(self.run_cfg)
        if self.kfold.method.endswith("gkf"):
            assert (
                self.kfold.columns != ""
            ), f"{self.kfold.columns} should not be empty list when group kfold"
            kf_generator = kf.split(
                X_train, y_train, groups=X_train[self.kfold.columns],
            )
        else:
            kf_generator = kf.split(X_train, y_train)
        for fold_i, (tr_idx, val_idx) in enumerate(kf_generator):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            if self.run_cfg.model.name == "ModelLR":
                model = LRModel(self.run_cfg)
                model.train(X_tr, y_tr)
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelLGBM":
                model = LGBMModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            else:
                raise ValueError(f"{self.run_cfg.model.name} is not supported")
            oof_pred[val_idx] = y_pred[:, 1]
            models.append(model)
        if self.model.eval_metric == "auc":
            score = AUC(y_train, oof_pred)
        elif self.model.eval_metric == "pruac":
            score = PRAUC(y_train, oof_pred)
        elif self.model.eval_metric == "rmse":
            score = RMSE(y_train, oof_pred)
        else:
            raise ValueError(f"{self.model.eval_metric} is not supported")
        self.logger.info(f"cv score: {score:.6f}")
        results = ResultData(models=models, pred=oof_pred)
        Jbl.save(results, f"{ModelPath.model}/results_{self.run_name}.jbl")
        return results

    def _submit(
        self, models: List[Any], y_train: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        pred_test = np.zeros((len(test)))
        for fold_i in range(self.kfold.number):
            X_test = test.copy()
            pred = models[fold_i].predict(X_test.values)[:, 1]
            pred_test += pred / self.kfold.number
        sub = pd.DataFrame(pred_test, columns=[self.run_cfg.column.target])
        self.logger.info(pred_test)
        results = ResultData(models=None, pred=pred_test)
        Jbl.save(results, f"{ModelPath.model}/results_test_{self.run_name}.jbl")
        sub.to_csv(
            f"{ModelPath.submission}/submission_{self.run_name}.csv", index=False
        )

    def run_cv(self):
        feature = self.run_cfg.feature
        features: List[str] = [
            k for k in feature.__annotations__.keys() if getattr(feature, k)
        ]
        results_train: List[np.array] = [
            Jbl.load(f"{ModelPath.model}/results_{f}.jbl").pred for f in features
        ]
        results_test: List[np.array] = [
            Jbl.load(f"{ModelPath.model}/results_test_{f}.jbl").pred for f in features
        ]
        X_train = np.array(results_train).T
        X_train = pd.DataFrame(X_train, columns=[f"result_{f}" for f in features])
        X_test = np.array(results_test).T
        X_test = pd.DataFrame(X_test, columns=[f"result_{f}" for f in features])
        train = self._load().train
        y_train = train["target"]

        results = self._model(X_train, y_train)
        if self.run_cfg.model.name == "ModelLGBM":
            self._feature_importance(results.models)
        self._submit(results.models, y_train, X_test)


class AdversarialValidationRunner(TrainRunner):
    def _model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ResultData:
        models = []
        oof_pred = np.zeros((len(y_train)))
        kf = generate_kf(self.run_cfg)
        if self.kfold.method.endswith("gkf"):
            assert (
                self.kfold.columns != ""
            ), f"{self.kfold.columns} should not be empty list when group kfold"
            kf_generator = kf.split(
                X_train, y_train, groups=X_train[self.kfold.columns],
            )
        else:
            kf_generator = kf.split(X_train, y_train)
        for fold_i, (tr_idx, val_idx) in enumerate(kf_generator):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            X_tr = self._preprocess(X_tr, fold_i)
            X_val = self._preprocess(X_val, fold_i)
            if self.run_cfg.model.name == "ModelLGBM":
                model = LGBMModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelLGBMOptuna":
                model = LGBMOptunaModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelXGB":
                model = XGBModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            elif self.run_cfg.model.name == "ModelCBClf":
                model = CatClassifierModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict_proba(X_val, is_binary=True)
            elif self.run_cfg.model.name == "ModelCBReg":
                model = CatRegressorModel(self.run_cfg)
                model.train(X_tr, y_tr, X_val, y_val)
                y_pred = model.predict(X_val)
            else:
                raise ValueError(f"{self.run_cfg.model.name} is not supported")
            oof_pred[val_idx] = y_pred[:, 1]
            models.append(model)
        if self.model.eval_metric == "auc":
            score = AUC(y_train, oof_pred)
        elif self.model.eval_metric == "pruac":
            score = PRAUC(y_train, oof_pred)
        elif self.model.eval_metric == "rmse":
            score = RMSE(y_train, oof_pred)
        else:
            raise ValueError(f"{self.model.eval_metric} is not supported")
        self.logger.info(f"cv score: {score:.6f}")
        results = ResultData(models=models, pred=oof_pred)
        # Jbl.save(results, f"{ModelPath.model}/results_{self.run_name}.jbl")
        return results

    def run_cv(self):
        suffix = "_debug" if self.run_cfg.basic.is_debug else ""
        if check_exist(f"{DataPath.interim.train}{suffix}.jbl") and check_exist(
            f"{DataPath.interim.test}{suffix}.jbl"
        ):
            train = Jbl.load(f"{DataPath.interim.train}{suffix}.jbl")
            test = Jbl.load(f"{DataPath.interim.test}{suffix}.jbl")
        else:
            train_data = self._load()
            train = train_data.train
            test = train_data.test
            del train_data
            Jbl.save(train, f"{DataPath.interim.train}{suffix}.jbl")
            Jbl.save(test, f"{DataPath.interim.test}{suffix}.jbl")
        train = train.assign(is_train=1)
        test = test.assign(is_train=0)
        df = pd.concat([train.drop("target", axis=1), test], axis=0)
        df = df.sample(frac=1, random_state=42)
        X_train = df.drop("is_train", axis=1)
        y_train = df["is_train"]

        results = self._model(X_train, y_train)
        if self.run_cfg.model.name == "ModelLGBM":
            self._feature_importance(results.models)
