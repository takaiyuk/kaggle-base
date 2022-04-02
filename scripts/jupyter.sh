#!/bin/bash
SERVER_INFO=($(jupyter server list | sed -n 2P))
URL=${SERVER_INFO[0]}
echo $URL | sed "s/http:\/\/.*:8888/http:\/\/localhost:8888/g"
