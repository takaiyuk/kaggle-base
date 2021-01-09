#!/bin/bash

function checkdir () {
    if [ $(ls -1 | wc -l) -ne 1 ]; then
        exit 1
    fi
}

function makedirs () {
    mkdir -p docker

    mkdir -p docs
    mkdir -p docs/references

    mkdir -p input

    mkdir -p notebooks

    mkdir -p output/confusion
    mkdir -p output/feature
    mkdir -p output/importance
    mkdir -p output/log
    mkdir -p output/model
    mkdir -p output/optuna
    mkdir -p output/submission

    mkdir -p scripts
    mkdir -p scripts/kaggle

    mkdir -p src/data
    mkdir -p src/exp
    mkdir -p src/fe
    mkdir -p src/features
    mkdir -p src/models
    mkdir -p src/utils
}

function touch_keep () {
    touch docker/.gitkeep

    touch docs/.gitkeep
    touch docs/references/.gitkeep

    touch input/.gitkeep

    touch notebooks/.gitkeep

    touch output/confusion/.gitkeep
    touch output/feature/.gitkeep
    touch output/importance/.gitkeep
    touch output/log/.gitkeep
    touch output/model/.gitkeep
    touch output/optuna/.gitkeep
    touch output/submission/.gitkeep

    touch scripts/.gitkeep
    touch scripts/kaggle/.gitkeep

    touch src/data/.gitkeep
    touch src/exp/.gitkeep
    touch src/fe/.gitkeep
    touch src/features/.gitkeep
    touch src/models/.gitkeep
    touch src/utils/.gitkeep
}

function touch_init () {
    touch src/__init__.py
    touch src/data/__init__.py
    touch src/exp/__init__.py
    touch src/fe/__init__.py
    touch src/features/__init__.py
    touch src/models/__init__.py
    touch src/utils/__init__.py

    touch docker/pull.sh
    touch docker/run.sh
    touch docker/exec.sh
    touch docker/kill.sh

    touch docs/competition.md
    touch docs/log.md

    touch scripts/kaggle/download.sh
    touch scripts/kaggle/submit.sh
    touch scripts/jupyter.sh
    touch scripts/kaggle-utils.sh

    touch src/data/load.py
    touch src/models/base.py
    touch src/models/evaluate.py
    touch src/models/model_cb.py
    touch src/models/model_lgbm.py
    touch src/models/model_lr.py
    touch src/models/model_mlp.py
    touch src/models/model_ridge.py
    touch src/models/model_xgb.py
    touch src/utils/config.py
    touch src/utils/file.py
    touch src/utils/joblib.py
    touch src/utils/logger.py
    touch src/utils/memory.py
    touch src/kfold.py
    touch src/types.py

    touch .flake8
    touch .gitignore
    touch .isort.cfg
    touch README.md
    touch run.py
    touch transform-kaggle-notebook.sh
}

function chmod_shell () {
    chmod +x ./docker/exec.sh
    chmod +x ./docker/pull.sh
    chmod +x ./docker/run.sh
    chmod +x ./docker/kill.sh

    chmod +x ./scripts/kaggle/download.sh
    chmod +x ./scripts/kaggle/submit.sh
    chmod +x ./scripts/jupyter.sh
    chmod +x ./scripts/kaggle-utils.sh

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
    echo "input/*" >> .gitignore
    echo "kaggle_utils/" >> .gitignore
    echo "notebooks/*" >> .gitignore
    echo "output/confusion/*" >> .gitignore
    echo "output/feature/*" >> .gitignore
    echo "output/importance/*" >> .gitignore
    echo "output/log/*" >> .gitignore
    echo "output/model/*" >> .gitignore
    echo "output/optuna/*" >> .gitignore
    echo "output/submission/*" >> .gitignore
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

# checkdir
# makedirs
touch_keep
# touch_init
# chmod_shell
# gitignore_init
# flake8_init
# isort_init
# git_init
