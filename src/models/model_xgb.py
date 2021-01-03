from typing import Optional

import pandas as pd
import xgboost as xgb

from src.models import Model


class XGBModel(Model):
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
