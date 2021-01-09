#!/bin/bash

# 引数チェック
if [ $# != 2 ]; then
    echo "[ERROR] There must be two arguments for fe_config and run_config: e.g. fe000 run000"
    exit 1
fi

# src/hoge/ごとにブロックを作成する
function module_block () {
    n=$(echo $1 | sed -e 's/src\///g' | sed -e 's/\///g' | sed -e 's/.py//g')
    NAME=$(echo ${n^^})
    
    echo "" >> ./kaggle-notebook.py \
    && echo "############################################################" >> ./kaggle-notebook.py \
    && echo "# ${NAME}" >> ./kaggle-notebook.py \
    && echo "############################################################" >> ./kaggle-notebook.py \
    && echo "" >> ./kaggle-notebook.py
}

# 初期化
echo > ./kaggle-notebook.py
module_block IMPORT

# src/直下のファイルを展開
FILES=$(echo $(ls -d src/*.py))
for f in ${FILES}
do
    if [ ! "`echo ${f} | grep __init__.py`" ]; then
        module_block ${f}
    fi
    cat ${f} >> ./kaggle-notebook.py
done

# src/*/以下(src/fe, src/exp除く)のファイルを展開
DIRS=$(echo $(ls -d src/*/))
for d in ${DIRS}
do
    if [ ! "`echo ${d} | grep __pycache__`" ] && [ ! "`echo ${d} | grep fe`" ] && [ ! "`echo ${d} | grep exp`" ]; then 
        module_block ${d}
        FILES=$(echo $(ls -d ${d}*))
        for f in ${FILES}
        do
            if [ ! "`echo ${f} | grep __pycache__`" ]; then 
                cat ${f} >> ./kaggle-notebook.py
            fi
        done
    fi
done

# src/fe/, src/exp以下のファイルを展開
module_block CONFIG
cat src/fe/$1*.py >> ./kaggle-notebook.py
cat src/exp/$2*.py >> ./kaggle-notebook.py

# mainファイルを展開
module_block RUN
cat run.py >> ./kaggle-notebook.py
