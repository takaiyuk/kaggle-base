# kaggle-${COMPETITION_NAME}

## Usage

### Setup

```
$ cp .env.example .env
$ poetry install
$ poetry run inv kaggle-download
```

### Run

```
$ poetry run inv run -v verXXX
```

### Format

```
$ poetry run inv format
```

### Lint

```
$ poetry run inv lint
```

### Jupyter

```
$ poetry run inv jupyter
```

### Submission

```
$ poetry run inv kaggle-submit -v verXXX
```

## Directory Structure

```
.
├── data
├── docker
│  └── Dockerfile
├── input
│  └── ${COMPETITION_NAME}
├── notebooks
├── output
│  └── submissions
├── poetry.lock
├── pyproject.toml
├── README.md
├── src
│  ├── __init__.py
│  ├── ver001
│  │  ├── __init__.py
│  │  └── __main__.py
│  ├── filepath.py
│  └── utils.py
├── tasks.py
└── tests
   └── __init__.py
```
