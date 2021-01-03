from dataclasses import asdict
from typing import List

import pandas as pd

from src.config import BaseFeConfig
from src.features import FeatureData, encode, transform
from src.utils import Jbl, check_exist


def make_features(
    train: pd.DataFrame, fe_cfg: BaseFeConfig, prefix: str = None, suffix: str = None
):
    feature_list: List[str] = [
        k for k, v in asdict(fe_cfg.feature).items() if v is True
    ]
    for feature_str in feature_list:
        filepath = feature_str
        if prefix is not None:
            filepath = f"{prefix}/{filepath}"
        if suffix is not None:
            filepath = f"{filepath}_{suffix}ss"
        filepath += ".jbl"

        if check_exist(filepath):
            print(f"skip {feature_str}")
            continue
        print(f"run {feature_str}")
        if feature_str.endswith("Transformer"):
            t = getattr(transform, feature_str)
            t_ = t()
            t_.fit(train)
            train_agg = t_.get_features()
            key = t_.get_keys()
        elif feature_str.endswith("Encoder"):
            t = getattr(encode, feature_str)
            t_ = t()
            train_agg = t_.get_features()
            key = t_.get_categorical_features()
        else:
            raise ValueError(f"{feature_str} is not supported")
        Jbl.save(FeatureData(key=key, df=train_agg), filepath)
