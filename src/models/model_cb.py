from typing import Optional

import catboost as cb
import pandas as pd

from src.models import Model


class CatClassifierModel(Model):
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


class CatRegressorModel(Model):
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
