#!/bin/bash
source ./.env

CONTAINER_NAME=$COMPETITION_NAME
PORT=$1
if [[ $PORT == "" ]]; then
  PORT="8888"
fi
echo "PORT: ${PORT}"
docker run --gpus all -it -d --name ${CONTAINER_NAME} --shm-size=2048m -p ${PORT}:${PORT} -v ${PWD}:/${CONTAINER_NAME} -v ${HOME}/.kaggle:/root/.kaggle/ gcr.io/kaggle-gpu-images/python:${KAGGLE_IMAGE_TAG}
