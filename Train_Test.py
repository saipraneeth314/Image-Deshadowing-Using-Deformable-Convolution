import os
import torch

import utils

from tqdm import tqdm

def train_model(model, dataloaders, criterion_l1, criterion_robustloss, optimizer, scheduler, num_epochs, device, save_dir):
    # best_metric = float('-inf')  # Initialize the best metric to a very low value ,for Psnr,ssim
    best_metric = float('inf') # for rmse
    best_epoch = -1

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_metric = 0.0

            # Use tqdm to display progress bar
            data_loader = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Batch')

            psnr_non, ssim_non, rmse_non = 0,0,0
            psnr_shadow, ssim_shadow, rmse_shadow = 0,0,0

            # Iterate over data
            for images, masks, targets, img_names in data_loader:
                images = images.to(device)
                masks = masks.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    images+=masks
                    # images -= masks
                    outputs, features1, features2 = model(images)
                    
                    # loss=0.6*criterion_l1(features1, features2)+criterion_l1(outputs, targets)+10*(criterion_l1(features1, features2)*criterion_l1(outputs, targets))
                    # loss = criterion_l1(outputs, targets) + criterion_robustloss(features1, features2)
                    loss = criterion_l1(outputs, targets) + 0.4*utils.dice_coefficient(features1, features2)
                    # loss = criterion_l1(outputs, targets) + criterion_robustloss(features1, features2)
                    # loss = criterion_l1(outputs, targets) + utils.dice_coefficient(features1, features2)
                    loss = loss.mean() 
                    

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    for img in img_names:
                        index = img_names.index(img)
                        gen_img = utils.tensor2img(outputs.detach()[index].unsqueeze(1))
                        utils.save_img(gen_img, os.path.join('./generated_images', img_names[index]))
                        gen_img = utils.tensor2img(targets.detach()[index].unsqueeze(1))
                        utils.save_img(gen_img, os.path.join('./generated_images', 'target'+img_names[index]))

                # PSNR metric for complete_image, non-shadow region and shadow region, respectively
                # batch_metric = utils.psnr_np(outputs.detach(), targets.detach())
                # psnr_non += utils.psnr_np((outputs*(1-masks)).detach(), (targets*(1-masks)).detach())
                # psnr_shadow += utils.psnr_np((outputs*masks).detach(), (targets*masks).detach())


                # RMSE metric for complete_image, non-shadow region and shadow region, respectively
                # use this 
                batch_metric = utils.get_rmse(outputs.detach()*255, targets.detach()*255)
                rmse_non += utils.get_rmse((outputs*(1-masks)).detach()*255, (targets*(1-masks)).detach()*255)
                rmse_shadow += utils.get_rmse((outputs*masks).detach()*255, (targets*masks).detach()*255)
                        
                        

                # SSIM metric for complete_image, non-shadow region and shadow region, respectively
                # batch_metric = utils.ssim(outputs.detach(), targets.detach())
                # ssim_non += utils.ssim((outputs*(1-masks)).detach(), (targets*(1-masks)).detach())
                # ssim_shadow += utils.ssim((outputs*masks).detach(), (targets*masks).detach())


                running_loss += loss.item() * images.size(0)
                running_metric += batch_metric.item() * images.size(0)

                # Update tqdm progress bar
                data_loader.set_postfix(loss=loss.item(), metric=batch_metric.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric = running_metric / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, Metric: {epoch_metric:.4f}')
            # PSNR
            # print(f'Non_shadow_PSNR: {round(psnr_non.item()/len(dataloaders[phase]), 4)} || '
            #       f'shadow_PSNR: {round(psnr_shadow.item()/len(dataloaders[phase]), 4)}')
            # RMSE
            print(f'Non_shadow_rmse: {round(rmse_non.item()/len(dataloaders[phase]), 4)} || '
                  f'shadow_rmse: {round(rmse_shadow.item()/len(dataloaders[phase]), 4)}')
            
            # SSIM
            # print(f'Non_shadow_ssim: {round(ssim_non.item()/len(dataloaders[phase]), 4)} || '
            #       f'shadow_ssim: {round(ssim_shadow.item()/len(dataloaders[phase]), 4)}')

             # Rmse the best value is the epoch metric which is less
            if (phase == 'val' and epoch_metric < best_metric) or (phase == 'train' and epoch in [25, 50, 75, 99]):
            # if (phase == 'val' and epoch_metric > best_metric) or (phase == 'train' and epoch in [25, 50, 75, 99]):
                if phase == 'val':
                    print("Best model found! Saving...")
                    best_metric = epoch_metric
                    best_epoch = epoch

                # Save model, optimizer state_dict & epoch number
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                save_str = '_best' if phase == 'val' else f'_{epoch}'
                save_checkpoint(checkpoint, save_dir, save_str)

        
        # Update the learning rate scheduler
        if scheduler:
            scheduler.step()
    print(f"Training complete. Best model found in epoch {best_epoch + 1}.")

def save_checkpoint(checkpoint, save_dir, save_str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'checkpoint' + save_str + '.pth')
    torch.save(checkpoint, save_path)