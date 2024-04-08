import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
import math
import torch.nn.functional as F

import torch
from torch import nn
from torch import optim

# my create mask for kspace
def create_mask(shape, undersampling_factor, lambda_poisson):
    height = shape[1]
    width = shape[2]
    mask_shape = (height, width)
    # take the center of the k-space
    mask1 = np.zeros(mask_shape)
    side_factor = np.sqrt(undersampling_factor)
    width_middle = width // 2
    height_middle = height // 2
    left = int(width_middle - width_middle / side_factor)
    right = int(width_middle + width_middle / side_factor)
    upper = int(height_middle - height_middle / side_factor)
    lower = int(height_middle + height_middle / side_factor)
    mask1[upper:lower, left:right] = 1

    # add a poisson disk sampling
    mask2 = np.random.poisson(lam=lambda_poisson, size=mask_shape)
    # make msk2 binary
    mask2[mask2 > 0] = 1
    
    # logical or mask1 and mask2
    mask = np.logical_or(mask1, mask2)

    return mask

# input spacial domain, output spacial domain
# convert to k-space, apply mask, convert back to spacial domain
class underSamplingOperator(nn.Module):
    def __init__(self, mask_factor, mask_lambda, device=torch.device("cuda:0")):
        super(underSamplingOperator, self).__init__()
        self.mask_factor = mask_factor
        self.mask_lambda = mask_lambda
        
    def AHA(self, x): # x is in 3 channels 
        # convert x to gray scale
        grayscale_transform  = T.Grayscale(num_output_channels=1)
        x = grayscale_transform(x)
        x = x.squeeze(1)
        
        # create mask
        mask = create_mask(x.shape, self.mask_factor, self.mask_lambda)
        # convert to tensor
        mask = torch.cuda.FloatTensor(mask)

        # x is a tensor 
        # convert to k-space
        x_k = torch.fft.fft2(x)
        x_k = torch.fft.fftshift(x_k)
        x_k_u = x_k * mask

        # convert back to image space
        x_u = torch.fft.ifft2(x_k_u)
        x_u = torch.abs(x_u)
        
        # 1 channel to 3 channel
        x_u_3 = x_u.repeat(3, 1, 1)
        x_u_3 = x_u_3.unsqueeze(0)

        return x_u_3
    

# conjugate gradient block
class cg_block(nn.Module):
    def __init__(self, cgIter, cgTol):
        super(cg_block, self).__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol
        
    def forward(self, lhs, rhs, x0):
        fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3]))
        x = x0
        r = rhs-lhs(x0)
        p = r
        rTr = fn(r,r)
        eps=torch.tensor(1e-10)
        for i in range(self.cgIter):
            Ap = lhs(p)
            alpha=rTr/(fn(p,Ap)+eps)
            x = x +  alpha[:,None,None,None] * p
            r = r -  alpha[:,None,None,None] * Ap
            rTrNew = fn(r,r)
            if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
                break
            beta = rTrNew / (rTr+eps)
            p = r + beta[:,None,None,None] * p
            rTr=rTrNew
        return x
    


