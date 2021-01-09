import sys
from typing import Optional, Union

import cudf
import pandas as pd

from src.types import InputPath


class Loader:
    def __init__(self, input_path: InputPath = InputPath) -> None:
        self.path = input_path

    def _load_csv(
        self, path: str, is_cudf: bool, **args
    ) -> Union[pd.DataFrame, cudf.DataFrame]:
        try:
            cdf = cudf.read_csv(path, **args)
            if is_cudf:
                return cdf
            else:
                return cdf.to_pandas()
        except Exception:
            print("use pandas")
            return pd.read_csv(path, **args)

    def train(
        self, path: Optional[str] = None, is_cudf: bool = False, **args
    ) -> Union[pd.DataFrame, cudf.DataFrame]:
        if path is None:
            path = self.path.train
        print(f"load {sys._getframe().f_code.co_name} path: {path}")
        return self._load_csv(path, is_cudf)

    def test(
        self, path: Optional[str] = None, is_cudf: bool = False
    ) -> Union[pd.DataFrame, cudf.DataFrame]:
        if path is None:
            path = self.path.test
        print(f"load {sys._getframe().f_code.co_name} path: {path}")
        return self._load_csv(path, is_cudf)

    def sub(self, path: Optional[str] = None) -> pd.DataFrame:
        if path is None:
            path = self.path.sub
        print(f"load {sys._getframe().f_code.co_name} path: {path}")
        return self._load_csv(path, is_cudf=False)
