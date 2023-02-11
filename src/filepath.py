from dataclasses import dataclass
from pathlib import Path

from src.utils import read_env


env_dict = read_env()
COMPETITION_NAME = env_dict["COMPETITION_NAME"]


@dataclass
class InputPath:
    prefix: Path = Path(f"input/{COMPETITION_NAME}")
    train: Path = prefix / "train.csv"
    test: Path = prefix / "test.csv"
    sample_submission: Path = prefix / "sample_submission.csv"


@dataclass
class InterimPathPrefix:
    prefix: Path = Path("data")
    train: Path = prefix / "train"
    test: Path = prefix / "test"


@dataclass
class OutputPathPrefix:
    prefix: Path = Path("output")
    submissions: Path = prefix / "submissions"
