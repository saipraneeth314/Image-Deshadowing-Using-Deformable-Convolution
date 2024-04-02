import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import torch.nn.functional as F

from torchvision.utils import make_grid
from torch.autograd import Variable
from math import exp
import math

# PSNR
#-----------------------------------------------
def psnr_np(pd, gt):
    # pd = pd.to(torch.float64)
    # gt = gt.to(torch.float64)
    mse = torch.mean((pd - gt)**2)
    if mse == 0:
        return float('inf')
    return torch.Tensor([20 * math.log10(1.0 / math.sqrt(mse))])
#-----------------------------------------------


# Dice - similarity
#--------------------------
def dice_coefficient(pred, target):
    smooth = 1e-6  # Small constant to avoid division by zero
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice
# --------------------------------

# Robust loss
#-----------------------------------------------
class robustloss(torch.nn.Module):
  """Rain Robust Loss"""
  def __init__(self):
    super(robustloss, self).__init__()
    # self.batch_size = batch_size
    self.n_views = 2
    self.temperature = 0.07
    # self.device = device
    # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

  def forward(self, features1,features2):
    # print(features1.device)
    features=torch.cat((features1,features2),dim=0)
    logits, labels = self.info_nce_loss(features)
    return F.cross_entropy(logits, labels)

  def info_nce_loss(self, features):
    labels = torch.cat([torch.arange(features.size(0)//2) for i in range(self.n_views)], dim=0)
    # print(labels.shape)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # print(labels.shape)
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    # print(mask.shape)
    labels = labels[~mask].view(labels.shape[0], -1)    
    # print(similarity_matrix.shape)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / self.temperature
    return logits,labels
#-----------------------------------------------


# SSIM Metric
#-----------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
#-----------------------------------------------


#RMSE
#-----------------------------------------------
def get_rmse(pd, gt):
    # pd=rgb2lab(pd,pd.device)
    # gt=rgb2lab(gt,gt.device)
    pd = torch.flatten(pd)
    gt= torch.flatten(gt)
    score = torch.sqrt(torch.mean((pd - gt) ** 2))

    return score
#-----------------------------------------------


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu()
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        # img_np = (img_np * 255.0).round()
        img_np = (img_np * 150.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def rgb2xyz(rgb, device): # rgb from [0,1]
    rgb = torch.abs(rgb)

    mask = (rgb > .04045).type(torch.FloatTensor)
    mask = mask.to(device)

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)


    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    return out

def xyz2lab(xyz, device):
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(device)
  
    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    mask = mask.to(device)

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    return out

def rgb2lab(rgb, device, ab_norm = 110.,l_cent = 50.,l_norm = 100.):
    lab = xyz2lab(rgb2xyz(rgb, device), device)
    l_rs = (lab[:,[0],:,:]-l_cent)/l_norm
    ab_rs = lab[:,1:,:,:]/ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    return out

