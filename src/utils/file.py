import os


def check_exist(filepath: str) -> bool:
    return os.path.exists(filepath)


def mkdir(filepath: str) -> bool:
    os.makedirs(filepath, exist_ok=True)
