#!/bin/bash
CONTAINER_NAME="kaggle-base"
docker start ${CONTAINER_NAME} && docker exec -it ${CONTAINER_NAME} /bin/bash
