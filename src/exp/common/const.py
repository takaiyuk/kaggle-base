from dataclasses import dataclass
from typing import Dict


def get_competition_name(path: str = ".env"):
    def read_env(path: str = ".env") -> Dict[str, str]:
        """Reads the .env file and returns a dict with the values"""
        with open(path) as f:
            return {line.split("=")[0]: line.split("=")[1].strip() for line in f}

    try:
        e = read_env(path)
    except FileNotFoundError:
        _COMPETITION_NAME = ""
        assert _COMPETITION_NAME != ""
        e = {"COMPETITION_NAME": _COMPETITION_NAME}
    competition_name = e.get("COMPETITION_NAME")
    if competition_name is None or competition_name == "":
        raise KeyError(f"COMPETITION_NAME in {path} is invalid: {competition_name}")
    return competition_name


COMPETITION_NAME = get_competition_name()


@dataclass(frozen=True)
class InputPath:
    _prefix: str = f"./input/{COMPETITION_NAME}"
    train: str = f"{_prefix}/train.csv"
    test: str = f"{_prefix}/test.csv"
    sample_submission: str = f"{_prefix}/sample_submission.csv"


@dataclass(frozen=True)
class OutputPath:
    _prefix: str = "./output"
    features: str = f"{_prefix}/features"
    importances: str = f"{_prefix}/importances"
    logs: str = f"{_prefix}/logs"
    models: str = f"{_prefix}/models"
    optuna: str = f"{_prefix}/optuna"
    submissions: str = f"{_prefix}/submissions"
