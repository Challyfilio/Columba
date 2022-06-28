#!/bin/bash

cd ..
#Use: bash train.sh resnet50 aircraft 0 20

NET=$1   # [resnet50,resnet18]
DS=$2    # datasets [aircraft,aircraft_similar]
OTL=$3   # only_train_linear [0:False,1:True]
EPOCH=$4 # epochs

DIR=output/${DS}/${NET}/${EPOCH}/train_linear_${OTL}
if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}. Skip this job"
else
  echo "Run this job and save the output to ${DIR}"
  python main.py \
    --output_dir ${DIR} \
    --net ${NET} \
    --dataset ${DS} \
    --only_train_linear ${OTL} \
    --epochs ${EPOCH}
fi
