from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge

from src.utils.models.base import BaseModel


class LRModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        params = self.params.copy()
        self.model = LR(params)
        self.model.fit(X_tr, y_tr)
        print(self.model.coef_)


class RidgeModel(BaseModel):
    def train(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        params = self.params.copy()
        self.model = Ridge(params)
        self.model.fit(X_tr, y_tr)
        print(self.model.coef_)
