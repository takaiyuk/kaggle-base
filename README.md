# [COMPETITION NAME]

[DESCRIPTION OF COMPETITION]

[URL OF COMPETITION]

## Setup

```
$ git clone git@github.com:takaiyuk/kaggle-base.git kaggle-$COMPETITION_NAME
$ cp .env.example .env  # Fill in 'COMPETITION_NAME'
$ ./scripts/initialize.sh

$ ./scripts/docker/run.sh
$ ./scripts/docker/exec.sh
root@xxxxx:/workspace# venv-activate
(venv) root@xxxxx:/workspace# ./scripts/kaggle/download.sh
(venv) root@xxxxx:/workspace# pip install -r requirements.txt
```

### Generate new exp directory

```
$ ./scripts/gen-new-exp.sh ${OLD_EXP_NAME_LIKE_EXP000}
```

## Run

```
$ ./scripts/docker/exec.sh
root@xxxxx:/workspace# venv-activate
(venv) root@xxxxx:/workspace# ./scripts/run.sh expXXX
```

### Run (Debug mode)

```
$ ./scripts/docker/exec.sh
root@xxxxx:/workspace# venv-activate
(venv) root@xxxxx:/workspace# ./scripts/run.sh expXXX debug
```

### Jupyter

```
$ ./scripts/docker/exec.sh
root@xxxxx:/workspace# venv-activate
(venv) root@xxxxx:/workspace# ./scripts/jupyter.sh
```
