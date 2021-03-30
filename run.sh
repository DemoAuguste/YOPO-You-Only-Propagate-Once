

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# CIFAR-10
# resnet18
cd $SHELL_FOLDER/experiments/CIFAR10/pre-res18.yopo-5-3/

for i in {1..5}
do
    /usr/bin/python3 train.py
    # save the result
    mkdir -p $SHELL_FOLDER/results/cifar10/res18/$i/
    cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar10/res18/$i/last.checkpoint
done

# resnet34
cd $SHELL_FOLDER/experiments/CIFAR10/wide34.yopo-5-3/
for i in {1..5}
do
    /usr/bin/python3 train.py
    # save the result
    mkdir -p $SHELL_FOLDER/results/cifar10/res34/$i/
    cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar10/res34/$i/last.checkpoint
done


# CIFAR-100
# resnet18
cd $SHELL_FOLDER/experiments/CIFAR100/pre-res18.yopo-5-3/

for i in {1..5}
do
    /usr/bin/python3 train.py
    # save the result
    mkdir -p $SHELL_FOLDER/results/cifar100/res18/$i/
    cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar100/res18/$i/last.checkpoint
done

# resnet34
cd $SHELL_FOLDER/experiments/CIFAR100/wide34.yopo-5-3/
for i in {1..5}
do
    /usr/bin/python3 train.py
    # save the result
    mkdir -p $SHELL_FOLDER/results/cifar100/res34/$i/
    cp ./log/models/last.checkpoint $SHELL_FOLDER/results/cifar100/res34/$i/last.checkpoint
done