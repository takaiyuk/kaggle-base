#!/bin/bash
COMPETITION_NAME=""  # Fill comepetiton name

RUNE_NAME=$1
echo ${RUNE_NAME}
FILEPATH=submissions/submission_${RUNE_NAME}.csv
MESSAGE=""
# message=$(cat ./outputs/${ymd}/${hms}/main.log | grep RMSE | sed 's/ //g' | sed 's/\t//g')
echo filepath: ${FILEPATH}
echo message: ${MESSAGE}

kaggle competitions submit -c ${COMPETITION_NAME} -f ${FILEPATH} -m ${MESSAGE}
