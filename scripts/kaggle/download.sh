#!/bin/bash
COMPETITION_NAME=""  # Fill comepetiton name

COMPETITION_PATH="input/${COMPETITION_NAME}"
if [ -n "$COMPETITION_NAME" ]; then
  echo ${COMPETITION_NAME}
  kaggle competitions download -c ${COMPETITION_NAME} -p .
  mkdir -p ${COMPETITION_PATH}
  unzip ${COMPETITION_NAME}.zip -d ${COMPETITION_PATH}
  rm ${COMPETITION_NAME}.zip
else
  echo "COMPETITION_NAME should be provided"
  exit 1
fi
