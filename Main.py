import os
import torch
import torch.nn as nn

from Model import ShadowRemoval
import utils
from Dataloader import ShadowDataset
from Train_Test import train_model


from torch.utils.data import DataLoader

def main():
    bs = 4 
    #im_size = 256
    # root_dir = '../ISTD_Dataset'
    # root_dir = '../ISTD_Adjusted' 
    root_dir = '../srd'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    num_epochs = 100

    model = ShadowRemoval()
    model = model.to(device)
    
    criterion_l1 = nn.L1Loss().to(device)
    criterion_robustloss = utils.robustloss().to(device)

    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    # optimizer = torch.optim.Adam(optim_params, lr=4e-4, betas=(0.9, 0.99))

    # This is for using the weight decay in the decoder part
    optimizer = torch.optim.Adam([
    {'params': [param for name,param in model.named_parameters() if 'decoder' not in name],'weight_decay':0.0},  # Parameters without weight decay
    {'params': model.Rnet1.decoder.parameters(), 'weight_decay': 1e-5} # Parameters with weight decay 
    ], lr=4e-4, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], gamma=0.5)

    min_lr = 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    -1, 
    eta_min = min_lr)
    
    train_dataset = ShadowDataset(os.path.join(root_dir, 'train'), is_train=True)
    val_dataset = ShadowDataset(os.path.join(root_dir, 'test'), is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                           num_workers=4,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False,drop_last=True)
    dataloaders = {'train': train_loader, 'val': val_loader}

    save_dir = './checkpoints'

    train_model(model, dataloaders, criterion_l1, criterion_robustloss, optimizer, scheduler, num_epochs, device, save_dir)    

if __name__ == '__main__':
    main()