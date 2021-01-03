import random
from collections import Counter, defaultdict
from typing import Generator, List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from src.config import BaseFeConfig, BaseRunConfig


class StratifiedGroupKFold:
    """https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation"""

    def __init__(self, n_splits: int, random_state: int = None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], groups: List[str]
    ) -> Generator:
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [
                        y_counts_per_fold[i][label] / y_distr[label]
                        for i in range(self.n_splits)
                    ]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices


def generate_kf(cfg: Union[BaseFeConfig, BaseRunConfig]) -> Generator:
    if cfg.kfold.method == "kf":
        kf = KFold(
            n_splits=cfg.kfold.number,
            shuffle=cfg.kfold.shuffle,
            random_state=cfg.basic.seed,
        )
    elif cfg.kfold.method == "skf":
        kf = StratifiedKFold(
            n_splits=cfg.kfold.number,
            shuffle=cfg.kfold.shuffle,
            random_state=cfg.basic.seed,
        )
    elif cfg.kfold.method == "gkf":
        kf = GroupKFold(n_splits=cfg.kfold.number)
    elif cfg.kfold.method == "sgkf":
        kf = StratifiedGroupKFold(
            n_splits=cfg.kfold.number, random_state=cfg.basic.seed
        )
    elif cfg.kfold.method == "tskf":
        kf = TimeSeriesSplit(n_splits=cfg.kfold.number)
    else:
        raise ValueError(f"{cfg.kfold.method} is not supported")
    return kf
