import argparse
import warnings
from typing import Union

from omegaconf import OmegaConf

from src import config
from src.config import BaseFeConfig, BaseRunConfig
from src.features.runner import FeatureRunner
from src.models.runner import StackingRunner, TrainRunner
from src.utils.logger import Logger

warnings.filterwarnings("ignore")


def run_feautres():
    fe_cfg = getattr(config, args.fe).FeConfig
    print_cfg(fe_cfg)
    FeatureRunner(fe_cfg).run_cv()


def run_models():
    run_cfg = getattr(config, args.run).RunConfig
    logger = Logger(run_name=run_cfg.basic.name)
    print_cfg(run_cfg)
    TrainRunner(run_cfg, logger).run_cv()


def run_stacking():
    run_cfg = getattr(config, args.run).RunConfig
    logger = Logger(run_name=run_cfg.basic.name)
    print_cfg(run_cfg)
    StackingRunner(run_cfg, logger).run_cv()


def print_cfg(cfg: Union[BaseFeConfig, BaseRunConfig]):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_fe = subparsers.add_parser("features", help="see `features -h`")
    parser_fe.add_argument("--fe", type=str, required=True, help="features config")
    parser_fe.set_defaults(func=run_feautres)
    parser_run = subparsers.add_parser("models", help="see `models -h`")
    parser_run.add_argument("--run", type=str, required=True, help="models config")
    parser_run.set_defaults(func=run_models)
    parser_stk = subparsers.add_parser("stacking", help="see `models -h`")
    parser_stk.add_argument("--run", type=str, required=True, help="stacking config")
    parser_stk.set_defaults(func=run_stacking)
    args = parser.parse_args()
    args.func()
