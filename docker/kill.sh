#!/bin/bash
CONTAINER_NAME="kaggle-base"
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
