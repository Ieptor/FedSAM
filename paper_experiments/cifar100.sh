#!/usr/bin/env bash

pushd ../models

declare -a alphas=("0")

function run_fedVC1000() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model resnet -lr 0.1 --weight-decay 0.0001 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --server-momentum 0.9 --num-workers 0 --where-loading init -alpha ${alpha} --sizeVC 256
}

function run_fedVC0() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model resnet -lr 0.1 --weight-decay 0 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha} --sizeVC 256
}


function run_fedVC05() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model resnet -lr 0.1 --weight-decay 0 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha} --sizeVC 256
}


function run_FedAVG1000() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model resnet -lr 0.1 --weight-decay 0.0001 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --server-momentum 0.9 --num-workers 0 --where-loading init -alpha ${alpha}
}

function run_FedAVG0() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model resnet -lr 0.1 --weight-decay 0 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}


function run_FedAVG05() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar100 --num-rounds 10000 --eval-every 100 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model resnet -lr 0.1 --weight-decay 0 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}


echo "####################### EXPERIMENTS ON CIFAR10 #######################"
for alpha in "${alphas[@]}"; do
  echo "Alpha ${alpha}"
  run_fedVC0 "${alpha}"
  run_FedAVG0 "${alpha}"
  echo "Done"
done