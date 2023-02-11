FROM python:3.10.8-bullseye as builder
ENV APP_NAME kaggle
ENV APP_HOME /workspace/$APP_NAME
ENV POETRY_HOME /opt/poetry
ENV PATH $POETRY_HOME/venv/bin/:$PATH
COPY poetry.lock pyproject.toml ./
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 -
RUN poetry config virtualenvs.create false \
    && poetry install --no-ansi --no-interaction --no-root
WORKDIR $APP_HOME
EXPOSE 8888
