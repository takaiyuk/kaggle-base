import os
import shutil


def check_exist(filepath: str) -> bool:
    return os.path.exists(filepath)


def mkdir(filepath: str) -> bool:
    os.makedirs(filepath, exist_ok=True)


def rmdir(directory: str) -> None:
    shutil.rmtree(directory)
