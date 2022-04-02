#!/bin/bash
source ./.env
USERNAME=takaiyuk

sudo rm -r .git
git init
git remote add origin git@github.com:$USERNAME/kaggle-$COMPETITION_NAME
