#!/bin/sh

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# CIFAR-10
# resnet18
echo $SHELL_FOLDER/experiments/CIFAR100/mobilenetv2.natural/
cd $SHELL_FOLDER/experiments/CIFAR100/mobilenetv2.natural/

for i in $(seq 1 5)
do
    python3 train.py
    # save the result
    mkdir -p $SHELL_FOLDER/results/cifar100/mobilenetv3/$i/
    cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar100/mobilenetv3/$i/natural.checkpoint
done

# # resnet34
# echo $SHELL_FOLDER/experiments/CIFAR10/wide34.yopo-5-3/
# cd $SHELL_FOLDER/experiments/CIFAR10/wide34.yopo-5-3/
# for i in {1..5}
# do
#     python3 train.py
#     # save the result
#     mkdir -p $SHELL_FOLDER/results/cifar10/res34/$i/
#     cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar10/res34/$i/last.checkpoint
# done


# # CIFAR-100
# # resnet18
# echo $SHELL_FOLDER/experiments/CIFAR100/pre-res18.yopo-5-3/
# cd $SHELL_FOLDER/experiments/CIFAR100/pre-res18.yopo-5-3/

# for i in {1..5}
# do
#     python3 train.py
#     # save the result
#     mkdir -p $SHELL_FOLDER/results/cifar100/res18/$i/
#     cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar100/res18/$i/last.checkpoint
# done

# # resnet34
# echo $SHELL_FOLDER/experiments/CIFAR100/wide34.yopo-5-3/
# cd $SHELL_FOLDER/experiments/CIFAR100/wide34.yopo-5-3/
# for i in {1..5}
# do
#     python3 train.py
#     # save the result
#     mkdir -p $SHELL_FOLDER/results/cifar100/res34/$i/
#     cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar100/res34/$i/last.checkpoint
# done