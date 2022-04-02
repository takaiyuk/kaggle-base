#!/bin/bash
OUTPUT_PATH=./src/kaggle-notebook.py

# 引数チェック
if [ $# != 1 ]; then
    echo "[ERROR] There must be one argument for experiment number: e.g. exp000"
    exit 1
fi

# src/hoge/ごとにブロックを作成する
function module_block () {
    n=$(echo $1 | sed -e 's/src\///g' | sed -e 's/\///g' | sed -e 's/.py//g')
    NAME=$(echo ${n^^})
    
    echo "" >> ${OUTPUT_PATH} \
    && echo "############################################################" >> ${OUTPUT_PATH} \
    && echo "# ${NAME}" >> ${OUTPUT_PATH} \
    && echo "############################################################" >> ${OUTPUT_PATH} \
    && echo "" >> ${OUTPUT_PATH}
}

# 初期化
echo > ${OUTPUT_PATH}

# src/utils/以下のファイルを展開
DIRS=$(echo $(ls -d src/utils/*/))
DIRS+=( "src/utils/" )
for d in ${DIRS[@]}
do
    if [ ! "`echo ${d} | grep __pycache__`" ]; then 
        module_block ${d}
        FILES=$(echo $(ls -d ${d}*))
        for f in ${FILES}
        do
            if [ "`echo ${f} | grep .py`" ] && [ ! "`echo ${f} | grep __pycache__`" ] && [ ! "`echo ${f} | grep __init__.py`" ]; then 
                cat ${f} >> ${OUTPUT_PATH}
            fi
        done
    fi
done

# src/exp???のファイルを展開
FILES=$(echo $(ls -d src/exp/${1}/*.py))
for f in ${FILES}
do
    if [ ! "`echo ${f} | grep __init__.py`" ]; then
        module_block ${f}
    fi
    cat ${f} >> ${OUTPUT_PATH}
done

# `from src.`を含む行を削除する
sed -i -e '/^from src\./d' ${OUTPUT_PATH}

# mainファイルを展開
module_block RUN
cat <<EOS >> ${OUTPUT_PATH}
if __name__ == "__main__":
    main()
EOS

# Import の整理
IMPORT_STATEMENTS=$(cat ${OUTPUT_PATH} | grep -e '^import ' -e '^from ')
sed -i -e '/^import /d' -e '/^from /d' ${OUTPUT_PATH}
cp ${OUTPUT_PATH} ./tmp
echo "$IMPORT_STATEMENTS" > ${OUTPUT_PATH}
cat ./tmp >> ${OUTPUT_PATH}
rm ./tmp
