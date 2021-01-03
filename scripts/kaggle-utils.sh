#!/bin/bash
function download () {
    sudo rm -r kaggle_utils
    git clone https://github.com/takaiyuk/kaggle_utils
}

function install () {
    apt-get update -y
    apt-get install -y --no-install-recommends libopencv-dev
    cd kaggle_utils
    pip install .
    cd ..
}

function uninstall () {
    pip uninstall kaggle_utils -y
}

if [ $1 = "download" ]; then
  download
elif [ $1 = "install" ]; then
  install
elif [ $1 = "uninstall" ]; then
  uninstall
else
   echo "$1 not supported"
   exit 0
fi


