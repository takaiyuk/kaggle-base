# COMPETITION TITLE

[COMPETITION URL]

## Setup & Run

### Training

```
$ ./scripts/download.sh
$ ./scripts/kaggle-utils.sh download
$ ./docker/pull.sh && ./docker/run.sh && ./docker/exec.sh
root@xxx:/workspace# venv-activate
(venv) root@xxx:/workspace# ./scripts/kaggle-utils.sh install
(venv) root@xxx:/workspace# ./scripts/jupyter.sh 8888
(venv) root@xxx:/workspace# python run.py fe -f 000
(venv) root@xxx:/workspace# python run.py exp -e 000
```

### Streamlit

```
$ ./docker/exec.sh
root@xxx:/workspace# venv-activate
(venv) root@xxx:/workspace# streamlit run main.py --browser.serverAddress localhost
```

## Docs

### Competition

https://github.com/takaiyuk/kaggle-base/blob/main/docs/competition.md

### Log

https://github.com/takaiyuk/kaggle-base/blob/main/docs/log.md
