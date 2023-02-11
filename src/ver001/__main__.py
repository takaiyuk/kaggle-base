import polars as pl
import structlog

from src.filepath import InputPath, InterimPathPrefix, OutputPathPrefix

logger = structlog.get_logger(__name__)
VERSION_NAME = "".join(__file__.split("/")[-2])


def main() -> None:
    logger.info(f"Run {VERSION_NAME}")
    if not InterimPathPrefix.train.joinpath("").exists():
        logger.info("Converting train raw data to interim data")
    if not InterimPathPrefix.test.joinpath("").exists():
        logger.info("Converting test raw data to interim data")
    df_sub = pl.read_csv(InputPath.sample_submission)
    df_sub.write_csv(OutputPathPrefix.submissions.joinpath(VERSION_NAME + ".csv"))


if __name__ == "__main__":
    main()
