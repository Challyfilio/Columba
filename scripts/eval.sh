#!/bin/bash

cd ..
#Use: bash eval.sh resnet50 aircraft 0 20

NET=$1   # [resnet50,resnet18]
DS=$2    # datasets [aircraft,aircraft_similar]
OTL=$3   # only_train_linear [0:False,1:True]
EPOCH=$4 # epochs

DIR=output/${DS}/${NET}/${EPOCH}/train_linear_${OTL}

python main.py \
  --output_dir ${DIR} \
  --net ${NET} \
  --dataset ${DS} \
  --only_train_linear ${OTL} \
  --epochs ${EPOCH} \
  --only_eval True
