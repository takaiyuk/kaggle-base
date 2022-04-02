#!/bin/bash
source ./.env

NUM_RESULTS=$1
if [ -n "$NUM_RESULTS" ]; then
  NUM_HEADERS=2
  NUM_HEADS=$(($NUM_RESULTS + $NUM_HEADERS))
  echo "NUM_HEADS: $NUM_HEADS"
  kaggle competitions submissions -c ${COMPETITION_NAME} | head -n $NUM_HEADS
else
  kaggle competitions submissions -c ${COMPETITION_NAME}
fi
