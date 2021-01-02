import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np


class Logger:
    """https://github.com/upura/ayniy/blob/master/ayniy/utils.py#L183"""

    def __init__(
        self,
        general_path: str = "logs/general.log",
        result_path: str = "logs/result.log",
        run_name: Optional[str] = None,
    ) -> None:
        os.makedirs("logs", exist_ok=True)
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")
        self.run_name = run_name
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        file_general_handler = logging.FileHandler(general_path)
        file_result_handler = logging.FileHandler(result_path)
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message: str) -> None:
        # 時刻をつけてコンソールとログに出力
        if self.run_name is not None:
            self.general_logger.info(
                f"[{self._now_string()}] [{self.run_name}] - {message}"
            )
        else:
            self.general_logger.info(f"[{self._now_string()}] - {message}")

    def result(self, message: str) -> None:
        self.result_logger.info(message)

    def result_ltsv(self, dic: dict):
        self.result(self._to_ltsv(dic))

    def result_scores(self, run_name: str, scores: List[float]) -> None:
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic["name"] = run_name
        dic["score"] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f"score_{i}"] = score
        self.result(self._to_ltsv(dic))

    def _now_string(self) -> str:
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _to_ltsv(self, dic: Dict[str, Any]) -> str:
        return "\t".join([f"{k}: {v}" for k, v in dic.items()])
