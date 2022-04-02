#!/bin/bash
source ./.env

SLEEP_SEC=10
RUN_NAME=$1
echo ${RUN_NAME}
FILEPATH=output/submissions/submission_${RUN_NAME}.csv
MESSAGE=$(cat ./output/logs/${RUN_NAME}/result.log | sed 's/ //g' | sed 's/\t//g' | sed -z 's/\n/, /g')
echo "filepath: ${FILEPATH}"
echo "message: ${MESSAGE}"

kaggle competitions submit -c ${COMPETITION_NAME} -f ${FILEPATH} -m "${MESSAGE}"
echo "sleep ${SLEEP_SEC}"
sleep $SLEEP_SEC
kaggle competitions submissions -c ${COMPETITION_NAME} | head -n 3
