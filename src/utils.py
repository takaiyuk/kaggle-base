import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional


def read_env(path: Path = Path(".env")) -> dict[str, str]:
    with open(path, "r") as f:
        return {line.split("=")[0]: line.split("=")[1].strip() for line in f.readlines()}


@contextmanager
def timer(name: str, logger: Optional[Any] = None) -> Generator:
    """
    Parameters
    ----------
    name : str
        the name of the function that measures time。
    logger: Optional[logging.Logger]
        logger if you want to use. If None, print() will be used.
    Examples
    --------
    >>> with timer("Process Modeling"):
            modeling()
    """
    t0 = time.time()
    yield
    message = f"[{name}] done in {(time.time() - t0)/60:.1f} min."
    if logger is not None:
        logger.info(message)
    else:
        print(message)
