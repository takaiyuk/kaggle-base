#!/bin/bash
PORT=$1
if [[ $PORT == "" ]]; then
  PORT="8888"
fi
echo "PORT: ${PORT}"
jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=$PORT
