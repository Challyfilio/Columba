#!/bin/bash

bash eval.sh resnet50 aircraft 0 0 100
bash eval.sh resnet50 aircraft 1 0 100
bash eval.sh resnet50 aircraft 1 1 100

bash eval.sh resnet18 aircraft 0 0 100
bash eval.sh resnet18 aircraft 1 0 100
bash eval.sh resnet18 aircraft 1 1 100

bash eval.sh resnet50 aircraft_similar 0 0 100
bash eval.sh resnet50 aircraft_similar 1 0 100
bash eval.sh resnet50 aircraft_similar 1 1 100

bash eval.sh resnet18 aircraft_similar 0 0 100
bash eval.sh resnet18 aircraft_similar 1 0 100
bash eval.sh resnet18 aircraft_similar 1 1 100