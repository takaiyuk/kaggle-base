from dataclasses import dataclass

COMPETITION_NAME = ""
assert COMPETITION_NAME != "", "fill in COMPETITION_NAME, not empty string"


@dataclass
class RawPath:
    prefix: str = f"./data/raw/{COMPETITION_NAME}"
    train: str = f"{prefix}/train.csv"
    test: str = f"{prefix}/test.csv"
    sample_submission: str = f"{prefix}/sample_submission.csv"


@dataclass
class InterimPath:
    prefix: str = "./data/interim"


@dataclass
class ProcessedPath:
    prefix: str = "./data/processed"


@dataclass
class DataPath:
    raw: RawPath = RawPath
    interim: InterimPath = InterimPath
    processed: ProcessedPath = ProcessedPath


@dataclass
class ModelPath:
    prefix: str = "./models"
    importance: str = f"{prefix}/importance"
    model: str = f"{prefix}/model"
    others: str = f"{prefix}/others"
    submission: str = "./submissions"
