from config import config
from dataset import create_test_dataset, create_train_dataset
from network import create_network

from training.train import eval_one_epoch, train_one_epoch
from pinecone import investigate_dataset, train_sensitive_data
from utils.misc import load_checkpoint

import torch.nn as nn
import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '--resume', default='log/models/last.checkpoint',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default:log/last.checkpoint)')
parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
parser.add_argument('-r','--ratio', type=float, default=0.01, help='ratio of sensitive data')
args = parser.parse_args()

print(args)

DEVICE = torch.device('cuda:{}'.format(args.d))
torch.backends.cudnn.benchmark = True

net = create_network()
net.to(DEVICE)

ds_val = create_test_dataset(512)
ds_train = create_train_dataset(128, shuffle=False)
total_num = len(ds_train.dataset)

TrainAttack = config.create_attack_method(DEVICE)
EvalAttack = config.create_evaluation_attack_method(DEVICE)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

if os.path.isfile(args.resume):
    load_checkpoint(args.resume, net)

for _ in range(5):
    total_counts = investigate_dataset(net, ds_train, DEVICE=DEVICE, eps=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], descrip_str='Investigating')

    # collect sensitive sample index.
    total_counts = total_counts.flatten()
    _, sort_idx = total_counts.sort()
    sensitive_idx = sort_idx[:int(args.ratio*total_num)]

    print('Evaluating -- Before fxing:')
    clean_acc, adv_acc = eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
    print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))

    print('--- The Pinecone Fixing Process ---')
    
    # evaluate sensitive layers.
    train_sensitive_data(net, ds_train, optimizer, sensitive_idx, DEVICE=DEVICE, AttackMethod=TrainAttack, descrip_str='Layer Investigating')
    print('Evaluating -- After training sensitive data:')
    clean_acc, adv_acc = eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
    print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))

    # adversarial training.
    train_one_epoch(net, ds_train, optimizer, nn.CrossEntropyLoss(), DEVICE,
                        'adversarial training', TrainAttack, adv_coef = 1.0)


    print('Evaluating -- After fxing:')
    clean_acc, adv_acc = eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
    print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))






