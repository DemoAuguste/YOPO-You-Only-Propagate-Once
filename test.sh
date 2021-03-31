#!/bin/sh

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

echo $SHELL_FOLDER/experiments/CIFAR10/pre-res18.yopo-5-3/
cd $SHELL_FOLDER/experiments/CIFAR10/pre-res18.yopo-5-3/

for i in {1..10}  
do  
echo $(expr $i \* 3 + 1);  
done  


for i in {1..10}  
do
    # python3 train.py
    # save the result
    echo $i
    mkdir -p $SHELL_FOLDER/results/cifar10/res18/$i/
    echo $SHELL_FOLDER/results/cifar10/res18/$i/last.checkpoint
    # cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar10/res18/$i/last.checkpoint
done