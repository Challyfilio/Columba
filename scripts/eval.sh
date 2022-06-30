#!/bin/bash

cd ..
#Use: bash eval.sh resnet50 aircraft 1 0 20

NET=$1   # [resnet50,resnet18]
DS=$2    # datasets [aircraft,aircraft_similar]
UPT=$3   # use_pretrain_model [0:False,1:True]
OTL=$4   # only_train_linear [0:False,1:True]
EPOCH=$5 # epochs

DIR=output/${DS}/${NET}/${EPOCH}/pretrain_model_${UPT}/train_linear_${OTL}

python main.py \
  --output_dir ${DIR} \
  --net ${NET} \
  --dataset ${DS} \
  --use_pretrain_model ${UPT} \
  --only_train_linear ${OTL} \
  --epochs ${EPOCH} \
  --only_eval True
