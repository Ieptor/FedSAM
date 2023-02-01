#!/usr/bin/env bash

pushd ../models

declare -a alphas=("1000")

function run_fedVC() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 5 -model resnet -lr 0.1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}


echo "####################### EXPERIMENTS ON CIFAR10 #######################"
for alpha in "${alphas[@]}"; do
  echo "Alpha ${alpha}"
  run_fedVC "${alpha}"
  echo "Done"
done