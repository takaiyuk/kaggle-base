#!/bin/bash
CONTAINER_NAME="kaggle-base"
sudo docker run --runtime=nvidia -it -d --name ${CONTAINER_NAME} -p 8501:8501 -p 8888:8888 -v ${PWD}:/workspace -v ${HOME}/.kaggle:/root/.kaggle/ takaiyuk/ml-table-gpu:latest
