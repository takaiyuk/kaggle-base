#!/bin/bash
source ./.env

CONTAINER_NAME=$COMPETITION_NAME
docker start ${CONTAINER_NAME} && docker exec -it ${CONTAINER_NAME} /bin/bash
