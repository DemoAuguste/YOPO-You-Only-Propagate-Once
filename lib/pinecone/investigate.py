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
            # print(label.size(), preds.size())
            label = label.reshape(-1,1)
            counts += preds.eq(label).cpu()
        
        if total_counts is None:
            total_counts = counts
        else:
            total_counts = torch.cat((total_counts, counts), axis=0)
        
    return total_counts


def eval_sensitive_layers(net, data_loader, indices, DEVICE=torch.device('cuda:0'), descrip_str='Layer Investigating'):
    sampler = torch.utils.data.SubsetRandomSampler(indices=indices)
    sample_dataloader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=512, shuffle=False, num_workers=2, sampler=sampler)

    for (inputs, targets) in sample_dataloader:
        print(inputs.size())
    

        

