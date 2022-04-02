import math
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional


def time_since(since: float, percent: float) -> str:
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)

    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{as_minutes(s)} (remain {as_minutes(rs)})"


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
