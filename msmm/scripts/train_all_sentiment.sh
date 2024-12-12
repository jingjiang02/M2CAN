#!/bin/bash
bash train.sh config_TumEmo.yaml $1
bash train.sh config_Twitter.yaml $1
bash train.sh config_Yelp.yaml $1
