from typing import Optional

import catboost as cb
import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd
import xgboost as xgb

from src.utils.models.base import BaseModel


class LGBMModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        # データのセット
        is_validation = X_val is not None
        lgb_train = lgb.Dataset(
            X_tr, y_tr, categorical_feature=self.categorical_features
        )
        if is_validation:
            lgb_eval = lgb.Dataset(
                X_val,
                y_val,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
            )

        # ハイパーパラメータの設定
        params = self.params.copy()
        if "num_boost_round" in params.keys():
            num_round = params.pop("num_boost_round")
        elif "n_estimators" in params.keys():
            num_round = params.pop("n_estimators")
        else:
            print(
                "[WARNING] num_round is set to 100: `num_boost_round` or `n_estimators` are not in the params"
            )
            num_round = 100

        # 学習
        if is_validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=500,
                early_stopping_rounds=early_stopping_rounds,
                **kwargs,
            )
        else:
            self.model = lgb.train(
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train],
                verbose_eval=500,
                **kwargs,
            )

    def feature_importance(
        self, fold_i: Optional[int] = None, importance_type: str = "gain"
    ) -> pd.DataFrame:
        df_fi = pd.DataFrame()
        df_fi["fold"] = fold_i
        df_fi["feature"] = self.model.feature_name()
        df_fi["importance"] = self.model.feature_importance(
            importance_type=importance_type
        )
        return df_fi


class LGBMOptunaModel(LGBMModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        # データのセット
        is_validation = X_val is not None
        lgb_train = optuna_lgb.Dataset(
            X_tr,
            y_tr,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        if is_validation:
            lgb_eval = optuna_lgb.Dataset(
                X_val,
                y_val,
                reference=lgb_train,
                categorical_feature=self.categorical_features,
                free_raw_data=False,
            )

        # ハイパーパラメータの設定
        params = self.params.copy()
        if "num_boost_round" in params.keys():
            num_round = params.pop("num_boost_round")
        elif "n_estimators" in params.keys():
            num_round = params.pop("n_estimators")
        else:
            print(
                "[WARNING] num_round is set to 100: `num_boost_round` or `n_estimators` are not in the params"
            )
            num_round = 100

        # 学習
        if is_validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = optuna_lgb.train(  # type: ignore
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=1000,
                early_stopping_rounds=early_stopping_rounds,
                **kwargs,
            )
        else:
            self.model = optuna_lgb.train(  # type: ignore
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train],
                verbose_eval=1000,
                **kwargs,
            )


class CatClassifierModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        # ハイパーパラメータの設定
        params = self.params.copy()
        self.model = cb.CatBoostClassifier(**params)

        # 学習
        self.model.fit(
            X_tr,
            y_tr,
            cat_features=self.categorical_features,
            eval_set=(X_val, y_val),
            verbose=500,
            use_best_model=True,
            plot=False,
        )


class CatRegressorModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        # ハイパーパラメータの設定
        params = self.params.copy()
        self.model = cb.CatBoostRegressor(**params)

        # 学習
        self.model.fit(
            X_tr,
            y_tr,
            cat_features=self.categorical_features,
            eval_set=(X_val, y_val),
            verbose=500,
            use_best_model=True,
            plot=False,
        )


class XGBModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        # データのセット
        is_validation = X_val is not None
        dtrain = xgb.DMatrix(X_tr, y_tr)
        if is_validation:
            dvalid = xgb.DMatrix(X_val, label=y_val)

        # ハイパーパラメータの設定
        params = self.params.copy()
        num_round = params.pop("num_round")

        # 学習
        if is_validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            watchlist = [(dtrain, "train"), (dvalid, "eval")]
            self.model = xgb.train(
                params,
                dtrain,
                num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            watchlist = [(dtrain, "train")]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)
