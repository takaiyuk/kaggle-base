from enum import Enum
from pathlib import Path

from invoke import task

from src.filepath import OutputPathPrefix
from src.utils import read_env

PROJECT_NAME = Path().absolute().name


@task
def echo(c):
    c.run("echo hello")


@task
def run(c, version):
    c.run(f"poetry run python -m src.{version}")


@task
def format(c):
    c.run("poetry run black .")
    c.run("poetry run isort .")


@task
def lint(c):
    c.run("poetry run black .")
    c.run("poetry run isort .")
    c.run("poetry run black --check --diff .")
    c.run("poetry run isort --check-only --diff .")
    c.run("poetry run mypy src")
    c.run("poetry run flake8 .")


@task
def jupyter(c):
    c.run("poetry run jupyter lab")


@task
def docker_build(c, image_name=None, sudo_option=False):
    if image_name is None:
        image_name = PROJECT_NAME
    sudo = "sudo" if sudo_option else ""
    arch = c.run("uname -m", hide=True).stdout.strip()
    if arch == "arm64":
        platform = "linux/arm64/v8"
    elif arch == "x86_64":
        platform = "linux/amd64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    c.run(f"{sudo} docker build --platform {platform} -t {image_name} -f ./docker/Dockerfile .", echo=True)


@task
def docker_run(c, image_name=None, sudo_option=False):
    if image_name is None:
        image_name = PROJECT_NAME
    sudo = "sudo" if sudo_option else ""
    arch = c.run("uname -m", hide=True).stdout.strip()
    if arch == "arm64":
        platform = "linux/arm64/v8"
    elif arch == "x86_64":
        platform = "linux/amd64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    c.run(
        f"{sudo} docker run --platform {platform} -it --rm -p 8888:8888 -v $PWD:/workspace/{image_name} -v ~/.kaggle:/root/.kaggle {image_name} /bin/bash",
        echo=True,
    )


@task
def kaggle_download(c):
    env = read_env()
    name = env["COMPETITION_NAME"]
    path = Path("input").joinpath(name)
    assert name != "", "$COMPETITION_NAME should be provided in .env"

    c.run(
        f"""
        poetry run kaggle competitions download -c {name} -p .
        mkdir -p {path}
        unzip {name}.zip -d {path}
        rm {name}.zip
    """
    )


@task
def kaggle_submit(c, version, message="", sleep=60):
    env = read_env()
    name = env["COMPETITION_NAME"]
    path = OutputPathPrefix.submissions.joinpath(version + ".csv")
    print(f"{name=} {path=} {message=}")

    c.run(
        f"""
        poetry run kaggle competitions submit -c {name} -f {path} -m "{message}"
        echo "\nsleep {sleep} sec. for waiting evaluation"
        sleep {sleep}
        poetry run kaggle competitions submissions -c {name} | head -n 3
    """
    )


@task
def kaggle_submission(c, num_results=None):
    env = read_env()
    name = env["COMPETITION_NAME"]
    pipe_string = ""
    if num_results:
        num_headers = 2
        try:
            num_heads = num_headers + int(num_results)
        except ValueError:
            raise ValueError(f"argument `num_results` must be integer: {num_results}")
        pipe_string = f"| head -n {num_heads}"

    c.run(
        f"""
        poetry run kaggle competitions submissions -c {name} {pipe_string}
    """
    )


@task
def kaggle_open(c, subdir=""):
    """

    Args:
        c:
        subdir: one of SubDir: 'overview', 'data', 'code', 'discussion', 'leaderboard', 'rules', 'team' or 'submissions'
    """

    class SubDir(Enum):
        overview = "overview"
        data = "data"
        code = "code"
        discussion = "discussion"
        leaderboard = "leaderboard"
        rules = "rules"
        team = "team"
        submissions = "submissions"

    env = read_env()
    name = env["COMPETITION_NAME"]
    if subdir == "":
        subdir = ""
    else:
        try:
            subdir = SubDir(subdir).value
        except ValueError:
            raise ValueError(f"argument `subdir` must be one of {[m.value for m in list(SubDir)]}: {subdir}")
    path = Path("https://www.kaggle.com/competitions").joinpath(name, subdir)
    c.run(f"open {str(path)}")
