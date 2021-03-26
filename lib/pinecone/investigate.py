import torch
from tqdm import tqdm
import torch.nn.functional as F
from .attack import *

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
            counts += preds.eq(label)
        
        if total_counts is None:
            total_counts = counts
        else:
            total_counts = torch.cat((total_counts, counts), axis=0)
        
    return total_counts




        

