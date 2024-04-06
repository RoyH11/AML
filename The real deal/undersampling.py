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
    # take the center of the k-space
    mask1 = np.zeros(shape)
    side_factor = np.sqrt(undersampling_factor)
    width_middle = shape[1] // 2
    height_middle = shape[0] // 2
    left = int(width_middle - width_middle / side_factor)
    right = int(width_middle + width_middle / side_factor)
    upper = int(height_middle - height_middle / side_factor)
    lower = int(height_middle + height_middle / side_factor)
    mask1[upper:lower, left:right] = 1

    # add a poisson disk sampling
    mask2 = np.random.poisson(lam=lambda_poisson, size=shape)
    
    # logical or mask1 and mask2
    mask = np.logical_or(mask1, mask2)
    
    return mask

class underSamplingOperator(nn.Module):
    def __init__(self, mask, device=torch.device("cuda:0")):
        self.mask = mask
        
    def AHA(self, x, mask):
        # x is a tensor 
        # convert to k-space
        x_k = torch.fft.fft2(x)
        x_k = torch.fft.fftshift(x_k)
        x_k_u = x_k * mask

        # convert back to image space
        x_u = torch.fft.ifft2(x_k_u)
        x_u = torch.abs(x_u)

        return x_u, x_k_u
    




# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# import torchvision.transforms as T
# from torchvision.io import read_image
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms.functional import InterpolationMode
# import math
# import torch.nn.functional as F

# import torch
# from torch import nn
# from torch import optim

# def bilinear_kernel(stride):
#     num_dims = len(stride)

#     shape = (1,) * num_dims
#     bilinear_kernel = torch.ones(*shape)

#     # The bilinear kernel is separable in its spatial dimensions
#     # Build up the kernel channel by channel
#     for channel in range(num_dims):
#         channel_stride = stride[channel]
#         kernel_size = 2 * channel_stride 
#         delta = torch.arange(1 - channel_stride, channel_stride+1)
#         channel_filter = (1 - torch.abs(delta / channel_stride))
#         # Apply the channel filter to the current channel
#         shape = [1] * num_dims
#         shape[channel] = kernel_size
#         bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
#         bilinear_kernel = bilinear_kernel/bilinear_kernel.sum()
#     return bilinear_kernel


# class underSamplingOperator(nn.Module):
#     def __init__(self, factor,device=torch.device("cuda:0")):
#         self.factor = factor
#         filt = bilinear_kernel([factor,factor])
#         self.kernel =  torch.zeros(3,3,filt.shape[0],filt.shape[1])
#         self.kernel[0,0] = filt
#         self.kernel[1,1] = filt
#         self.kernel[2,2] = filt
#         self.kernel = self.kernel.to(device)
        
#     def forward(self, x):
#         output = F.conv2d(x, self.kernel,stride=self.factor)
#         return output
        
#     def backward(self,x):
#         output = F.conv_transpose2d(x, self.kernel,stride=self.factor)
#         return output
        
#     def normal(self,x):
#         output = F.conv2d(x, self.kernel,stride=self.factor)
#         output = F.conv_transpose2d(output, self.kernel,stride=self.factor)
#         return output
    
# class cg_block(nn.Module):
#     def __init__(self, cgIter, cgTol):
#         super(cg_block, self).__init__()
#         self.cgIter = cgIter
#         self.cgTol = cgTol
        
#     def forward(self, lhs, rhs, x0):
#         fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3]))
#         x = x0
#         r = rhs-lhs(x0)
#         p = r
#         rTr = fn(r,r)
#         eps=torch.tensor(1e-10)
#         for i in range(self.cgIter):
#             Ap = lhs(p)
#             alpha=rTr/(fn(p,Ap)+eps)
#             x = x +  alpha[:,None,None,None] * p
#             r = r -  alpha[:,None,None,None] * Ap
#             rTrNew = fn(r,r)
#             if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
#                 break
#             beta = rTrNew / (rTr+eps)
#             p = r + beta[:,None,None,None] * p
#             rTr=rTrNew
#         return x