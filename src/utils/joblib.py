from typing import Any

import joblib


class Jbl:
    @staticmethod
    def load(filepath: str) -> Any:
        if not filepath.endswith(".jbl"):
            filepath += ".jbl"
        return joblib.load(filepath)

    @staticmethod
    def save(obj_: Any, filepath: str) -> None:
        if not filepath.endswith(".jbl"):
            filepath += ".jbl"
        joblib.dump(obj_, filepath, compress=3)
