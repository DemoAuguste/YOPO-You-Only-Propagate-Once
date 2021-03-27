import torch
# from torch.optim import optimizer
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from .attack import *
from collections import OrderedDict
from utils.misc import torch_accuracy, AvgMeter

def investigate_dataset(net, data_loader, DEVICE=torch.device('cuda:0'), eps=[0.0, 0.05, 0.1], descrip_str='Investigating'):
    """
    suppose the data loader iterates in a fixed order.
    """
    pbar = tqdm(data_loader)
    pbar.set_description(descrip_str)

    total_counts = None

    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        data.requires_grad = True
        output = net(data)
        # init_pred = output.max(1, keepdim=True)[1]

        loss = F.nll_loss(output, label)
        net.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        counts = torch.zeros((data.size(0), 1))

        for ep in eps:
            perturbed_data = fgsm_attack(data, ep, data_grad)
            adv_output = net(perturbed_data)
            preds = adv_output.max(1, keepdim=True)[1]
            # print(label.size(), preds.size())
            label = label.reshape(-1,1)
            counts += preds.eq(label).cpu()
        
        if total_counts is None:
            total_counts = counts
        else:
            total_counts = torch.cat((total_counts, counts), axis=0)
        
    return total_counts


def get_grad(model, y, y_true, optimizer, criterion):
    loss = criterion(y, y_true)
    optimizer.zero_grad()
    loss.backward()
    grad = OrderedDict()
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        grad[t2[0]] = t1.grad.data.clone()
    return grad


def set_grad(model, grad):
    for t1,t2 in zip(model.parameters(), model.state_dict().items()):
        if 'weight' in t2[0] and t2[0] in grad:
            t1.grad = grad[t2[0]]
        else:
            pass
            # print(t2[0])
            # print('something wrong.')


def get_grad_diff_layer_mask(grad, adv_grad, ratio=0.1):
    layer_mask = OrderedDict()
    avg_list = []

    def cal_mean_diff(g1, g2):
        diff = g1 - g2
        normalized_diff = (diff - torch.min(diff)) / (torch.max(diff) - torch.min(diff))
        return torch.mean(normalized_diff)

    for k,v in grad.items():
        if 'weight' not in k:
            continue
        layer_mask[k] = 0
        g1 = grad[k]
        g2 = adv_grad[k]
        avg_g = cal_mean_diff(g1, g2)
        avg_list.append(avg_g)
    
    # torch.kthvalue from smallest to largest.
    avg_list = torch.tensor(avg_list)
    threshold = torch.kthvalue(avg_list, int(avg_list.size(0) * (1 - ratio))).values
    for k,v in layer_mask.items():
        if v >= threshold:
            layer_mask[k] = 1

    return layer_mask


def generate_grad(grad, adv_grad, layer_mask=None, lr=0.01):
    ret_grad = OrderedDict()
    for k,v in grad.items():
        g1 = grad[k]
        g2 = adv_grad[k]
        if k in layer_mask:
            # g1 = grad[k]
            # g2 = adv_grad[k]

            diff = g1 - g2
            pos_mask = diff > 0
            pos_mask = pos_mask.int()
            neg_mask = diff < 0
            neg_mask = neg_mask.int()
            zero_mask = diff == 0
            zero_mask = zero_mask.int()

            sign_diff = g1.sign() + g2.sign()  # 0 represnets opposite search direction. 
            sign_diff[sign_diff>0] = 1
            sign_diff[sign_diff<0] = 1
            sign_diff[sign_diff == 0] = lr
            
            g1_part = pos_mask * g2 * sign_diff * (1 + lr)
            # g1_part = pos_mask * (g2 + diff * ratio)
            g2_part = neg_mask * g2 * sign_diff
            same_part = zero_mask * g2 

            ret_grad[k] = g1_part + g2_part + same_part
    
        else:
            ret_grad[k] = g2

    return ret_grad


def train_sensitive_data(net, data_loader, optimizer, indices, DEVICE=torch.device('cuda:0'), AttackMethod=None, descrip_str='Layer Investigating'):
    sampler = torch.utils.data.SubsetRandomSampler(indices=indices)
    sample_dataloader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=256, shuffle=False, num_workers=2, sampler=sampler)

    net.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(sample_dataloader)

    layer_mask = None

    for (inputs, label) in pbar:

        inputs = inputs.to(DEVICE)
        label = label.to(DEVICE)

        # original grads
        output = net(inputs)
        grad = get_grad(net, output, label, optimizer, criterion)

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, inputs, label)
            adv_output = net(adv_inp)
            adv_grad = get_grad(net, adv_output, label, optimizer, criterion)

        # Synthesize two gradients.
        layer_mask = get_grad_diff_layer_mask(grad, adv_grad, ratio=0.1)

        # ret_grad = generate_grad(grad, adv_grad, layer_mask=layer_mask)
        # set_grad(net, ret_grad)

        # optimizer.step()
    return layer_mask


def pinecone_train_one_epoch(net, batch_generator, optimizer,
                    criterion, DEVICE=torch.device('cuda:0'),
                    descrip_str='Training', AttackMethod = None, adv_coef = 1.0, layer_mask=None):
    '''

    :param attack_freq:  Frequencies of training with adversarial examples. -1 indicates natural training
    :param AttackMethod: the attack method, None represents natural training
    :return:  None    #(clean_acc, adv_acc)
    '''
    net.train()
    pbar = tqdm(batch_generator)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        pbar_dic = OrderedDict()
        TotalLoss = 0

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)
            optimizer.zero_grad()
            net.train()
            pred = net(adv_inp)

            adv_grad = get_grad(net, pred, label, optimizer, criterion)

            loss = criterion(pred, label)

            acc = torch_accuracy(pred, label, (1,))
            advacc = acc[0].item()
            advloss = loss.item()
            #TotalLoss = TotalLoss + loss * adv_coef
            # (loss * adv_coef).backward()


        pred = net(data)
        loss = criterion(pred, label)

        grad = get_grad(net, pred, label, optimizer, criterion)

        # layer_mask = get_grad_diff_layer_mask(grad, adv_grad, ratio=0.1)

        ret_grad = generate_grad(grad, adv_grad, layer_mask=layer_mask)
        set_grad(net, ret_grad)

        # loss.backward()

        optimizer.step()
        acc = torch_accuracy(pred, label, (1,))
        cleanacc = acc[0].item()
        cleanloss = loss.item()
        #pbar_dic['grad'] = '{}'.format(grad_mean)
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(advacc)
        pbar_dic['Advloss'] = '{:.2f}'.format(advloss)
        pbar.set_postfix(pbar_dic)
        


        
    

        

