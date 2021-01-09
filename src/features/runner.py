import warnings

from src.config import BaseFeConfig
from src.const import DataPath
from src.data import Loader
from src.features.preprocess import make_features
from src.models import TrainData
from src.models.kfold import generate_kf
from src.utils import Jbl, check_exist, mkdir, reduce_mem_usage

warnings.filterwarnings("ignore")


class FeatureRunner:
    def __init__(self, fe_cfg: BaseFeConfig):
        self.fe_cfg = fe_cfg
        self.kfold = fe_cfg.kfold

    def _load(self) -> TrainData:
        loader = Loader()
        if self.fe_cfg.basic.is_debug:
            train = loader.train(nrows=1_000_000)
        else:
            train = loader.train()
        test = loader.test()
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)
        return TrainData(train=train, test=test)

    def run_cv(self):
        suffix = "_debug" if self.fe_cfg.basic.is_debug else ""
        if check_exist(f"{DataPath.interim.train}{suffix}.jbl") and check_exist(
            f"{DataPath.interim.test}{suffix}.jbl"
        ):
            train = Jbl.load(f"{DataPath.interim.train}{suffix}.jbl")
        else:
            train_data = self._load()
            train = train_data.train
            test = train_data.test
            del train_data
            Jbl.save(train, f"{DataPath.interim.train}{suffix}.jbl")
            Jbl.save(test, f"{DataPath.interim.test}{suffix}.jbl")

        kf = generate_kf(self.fe_cfg)
        if self.kfold.method.endswith("gkf"):
            assert (
                self.kfold.columns != ""
            ), f"{self.kfold.columns} should not be empty list when group kfold"
            kf_generator = kf.split(
                train,
                train[self.fe_cfg.column.target].fillna(-1),
                groups=train[self.kfold.columns],
            )
        else:
            kf_generator = kf.split(train, train[self.fe_cfg.column.target].fillna(-1))
        for fold_i, (tr_idx, _) in enumerate(kf_generator):
            print(f"fold: {fold_i} - {tr_idx[:20]}")
            X_tr = train.iloc[tr_idx]

            prefix = f"features/{self.kfold.method}-{self.kfold.number}/fold_{fold_i}"
            suffix = None
            mkdir(prefix)
            make_features(
                X_tr, self.fe_cfg, prefix=prefix, suffix=suffix,
            )
