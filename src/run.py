import os
import warnings
from dataclasses import dataclass
from importlib import import_module

warnings.filterwarnings("ignore")


@dataclass
class RunParams:
    exp_name: str
    is_debug: bool


def run_exp(run_params: RunParams) -> None:
    exp_file_list = [e for e in os.listdir("src/exp") if run_params.exp_name in e]
    assert len(exp_file_list) == 1
    exp_module: str = os.path.splitext(exp_file_list[0])[0]  # [exp000] -> exp000
    module = import_module(f"src.exp.{exp_module}.main")
    print(f"execute main in src/exp/{exp_module}/main.py")
    module.main(run_params.is_debug)  # type: ignore
