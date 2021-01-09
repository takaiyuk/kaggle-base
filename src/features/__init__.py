from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class FeatureData:
    key: List[str]
    df: pd.DataFrame
