import sys
from typing import List, Optional

import cudf
import pandas as pd

from src.const import DataPath

train_dtype = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "int8",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "boolean",
}


class Loader:
    def __init__(self, data_path: DataPath = DataPath) -> None:
        self.path = data_path

    def _load_csv(self, path: str, is_cudf: bool = False, **args) -> pd.DataFrame:
        train_dtype_ = train_dtype.copy()
        if "usecols" in args.keys():
            usecols: List[str] = args.get("usecols")
            delcols = [key for key in train_dtype.keys() if key not in usecols]
            for col in delcols:
                del train_dtype_[col]
        try:
            cdf = cudf.read_csv(path, dtype=train_dtype_, **args)
            return cdf.to_pandas()
        except Exception:
            print("use pandas")
            return pd.read_csv(path, dtype=train_dtype_, **args)

    def train(
        self, path: Optional[str] = None, is_cudf: bool = False, **args
    ) -> pd.DataFrame:
        if path is None:
            path = self.path.raw.train
        print(f"load {sys._getframe().f_code.co_name} path: {path}")
        return self._load_csv(path)

    def test(self, path: Optional[str] = None, is_cudf: bool = False) -> pd.DataFrame:
        if path is None:
            path = self.path.raw.test
        print(f"load {sys._getframe().f_code.co_name} path: {path}")
        return self._load_csv(path)

    def sample_submission(
        self, path: Optional[str] = None, is_cudf: bool = False
    ) -> pd.DataFrame:
        if path is None:
            path = self.path.raw.example_sample_submission
        print(f"load {sys._getframe().f_code.co_name} path: {path}")
        return self._load_csv(path)
