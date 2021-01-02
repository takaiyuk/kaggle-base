#!/bin/bash
CONTAINER_NAME="kaggle-base"
sudo docker stop ${CONTAINER_NAME} && sudo docker rm ${CONTAINER_NAME}
