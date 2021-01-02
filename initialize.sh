#!/bin/bash

function checkdir () {
    if [ $(ls -1 | wc -l) -ne 1 ]; then
        exit 1
    fi
}

function makedirs () {
    mkdir -p data/external
    mkdir -p data/interim
    mkdir -p data/processed
    mkdir -p data/raw

    mkdir -p docker

    mkdir -p docs

    mkdir -p features

    mkdir -p models/importance
    mkdir -p models/model
    mkdir -p models/others

    mkdir -p notebooks

    mkdir -p references

    mkdir -p scripts

    mkdir -p src/config
    mkdir -p src/config/fe
    mkdir -p src/config/run
    mkdir -p src/data
    mkdir -p src/features
    mkdir -p src/models
    mkdir -p src/utils

    mkdir -p submissions
}

function touch_keep () {
    touch data/external/.gitkeep
    touch data/interim/.gitkeep
    touch data/processed/.gitkeep
    touch data/raw/.gitkeep

    touch docker/.gitkeep

    touch docs/.gitkeep

    touch features/.gitkeep
    
    touch models/importance/.gitkeep
    touch models/model/.gitkeep
    touch models/others/.gitkeep

    touch notebooks/.gitkeep

    touch references/.gitkeep

    touch scripts/.gitkeep

    touch src/config/.gitkeep
    touch src/config/fe/.gitkeep
    touch src/config/run/.gitkeep
    touch src/data/.gitkeep
    touch src/features/.gitkeep
    touch src/models/.gitkeep
    touch src/utils/.gitkeep

    touch submissions/.gitkeep
}

function touch_init () {
    touch src/__init__.py
    touch src/config/__init__.py
    touch src/config/fe/__init__.py
    touch src/config/fe/fe000.py
    touch src/config/run/__init__.py
    touch src/config/run/run000.py
    touch src/data/__init__.py
    touch src/features/__init__.py
    touch src/models/__init__.py
    touch src/utils/__init__.py

    touch docker/pull.sh
    touch docker/run.sh
    touch docker/exec.sh
    touch docker/kill.sh
    touch docs/competition.md
    touch docs/log.md
    touch src/config/base.py
    touch scripts/download.sh
    touch scripts/submit.sh
    touch src/const.py
    touch src/data/load.py
    touch src/features/preprocess.py
    touch src/features/runner.py
    touch src/models/base.py
    touch src/models/kfold.py
    touch src/models/runner.py
    touch .flake8
    touch .gitignore
    touch .isort.cfg
    touch README.md
    touch run.py
    touch transform-kaggle-notebook.sh
}

function touch_utils () {
    touch src/utils/file.py
    touch src/utils/joblib.py
    touch src/utils/logger.py
    touch src/utils/memory.py
}

function chmod_shell () {
    chmod +x ./docker/exec.sh
    chmod +x ./docker/pull.sh
    chmod +x ./docker/run.sh
    chmod +x ./docker/kill.sh

    chmod +x ./scripts/download.sh
    chmod +x ./scripts/submit.sh

    chmod +x ./transform-kaggle-notebook.sh
}

function gitignore_init () {
    wget -O .gitignore https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore
    echo "" >> .gitignore
    echo "# custom" >> .gitignore
    echo "*.csv" >> .gitignore
    echo "*.ipynb" >> .gitignore
    echo "*.jbl" >> .gitignore
    echo "*.png" >> .gitignore
    echo "data/external/*" >> .gitignore
    echo "data/interim/*" >> .gitignore
    echo "data/processed/*" >> .gitignore
    echo "data/raw/*" >> .gitignore
    echo "features/*" >> .gitignore
    echo "submissions/*" >> .gitignore
    echo ".vscode/" >> .gitignore
    echo "" >> .gitignore
    echo "# exclude" >> .gitignore
    echo "!.gitkeep" >> .gitignore
}

function flake8_init () {
    echo "[flake8]" >> .flake8
    echo "ignore = E203, E231, E266, E501, W503" >> .flake8
    echo "max-line-length = 80" >> .flake8
    echo "select = B,C,E,F,W,T4,B9" >> .flake8
    echo "exclude = .git, __pycache__, ./.venv/*" >> .flake8
}

function isort_init () {
    echo "[settings]" >> .isort.cfg
    echo "# must be same as .flake8" >> .isort.cfg
    echo "line_length=80" >> .isort.cfg
    echo "multi_line_output=3" >> .isort.cfg
    echo "include_trailing_comma=true" >> .isort.cfg
    echo "forced_separate=src" >> .isort.cfg
}

function git_init () {
    git init
}

checkdir
makedirs
touch_keep
touch_init
touch_utils
chmod_shell
gitignore_init
flake8_init
isort_init
git_init
