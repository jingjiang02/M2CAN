#!/bin/bash
bash train.sh config_AVA.yaml $1
bash train.sh config_PCCD.yaml $1
bash train.sh config_RPCD.yaml $1
