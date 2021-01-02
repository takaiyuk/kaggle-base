#!/bin/bash
CONTAINER_NAME="kaggle-base"
sudo docker start ${CONTAINER_NAME} && sudo docker exec -it ${CONTAINER_NAME} /bin/bash
