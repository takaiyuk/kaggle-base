#!/bin/bash
COMPETITION_NAME=""  # Fill comepetiton name

RUN_NAME=$1
echo ${RUNE_NAME}
FILEPATH=output/submission/submission_${RUN_NAME}.csv
MESSAGE=""
# message=$(cat ./outputs/${ymd}/${hms}/main.log | grep RMSE | sed 's/ //g' | sed 's/\t//g')
echo filepath: ${FILEPATH}
echo message: ${MESSAGE}

kaggle competitions submit -c ${COMPETITION_NAME} -f ${FILEPATH} -m ${MESSAGE}
