import json
from typing import Dict, List, Optional

import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd

from src.const import ModelPath
from src.models.base import Model


class LGBMModel(Model):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
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
        num_round = params.pop("num_boost_round")

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
            )
        else:
            self.model = lgb.train(
                params, lgb_train, num_round, valid_sets=[lgb_train], verbose_eval=500
            )


class LGBMOptunaModel(Model):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
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
        num_round = params.pop("num_boost_round")
        best_params: Dict = dict()
        tuning_history: List = list()

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
                best_params=best_params,
                tuning_history=tuning_history,
            )
        else:
            self.model = optuna_lgb.train(  # type: ignore
                params,
                lgb_train,
                num_round,
                valid_sets=[lgb_train],
                verbose_eval=1000,
                best_params=best_params,
                tuning_history=tuning_history,
            )
        with open(
            f"{ModelPath.others}/best_params_{self.run_name}_{self.fold_i}.json", "w"
        ) as f:
            json.dump(best_params, f, indent=4, separators=(",", ": "))
