#!/bin/bash
source ./.env

CONTAINER_NAME=$COMPETITION_NAME
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}
