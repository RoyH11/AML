import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
import math
import torch.nn.functional as F
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import h5py

class TrainDataset(Dataset):
  def __init__(self, image_dir, transform=None):
    super(TrainDataset, self).__init__()

    self.image_dir = image_dir
    self.transform = transform
              
    # preloading the data 
    h5_list = sorted(os.listdir(image_dir))

    # initialize the lists to store the data
    self.mri_images = []
    
    # open every h5 file and read the data
    for h5_file in h5_list:
      h5_path = os.path.join(image_dir, h5_file)
      with h5py.File(h5_path, 'r') as file:
        mri_data = file['kspace'][:]
        
        num_slices = mri_data.shape[0]
        num_coils = mri_data.shape[1]

        # get every slice combined with every coil
        for i in range(num_slices):
          combined_image = np.zeros_like(mri_data[0, 0])
          combined_image = combined_image.astype(np.float32)

          
          for j in range(num_coils):
            coil_k_space = mri_data[i, j]
            # perform inverse fourier transform
            coil_image = np.fft.ifft2(coil_k_space)
            coil_image = np.fft.fftshift(coil_image)
            coil_image = np.abs(coil_image)
            combined_image += coil_image**2
          
          combined_image = np.sqrt(combined_image)
          self.mri_images.append(combined_image)

  def __len__(self):
    return len(self.mri_images)
  
  def __getitem__(self, batch_index: int):
      if self.transform == None:
       mri_image = self.mri_images[batch_index]
      else:
        mri_image = self.transform(self.mri_images[batch_index])
      return mri_image
     

                    
                



# TODO: example code

# class TrainDataset(Dataset):
#   #Args:
#   #      image_dir (str): Train/Valid dataset address.
#   #      upscale_factor (int): Image up scale factor.
#   #      image_size (int): High resolution image size.
    
#     def __init__(self, image_dir, upscale_factor,transforms=None):
#         super(TrainDataset, self).__init__()

#         self.transforms = transforms
#         self.upscale_factor = upscale_factor
    
#         # Preloading the data 
#         filelist = sorted(os.listdir(image_dir))
#         image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in filelist]
#         self.images = []
#         for filename in image_file_names:
#             hr_image = read_image(filename)/255.
#             x = int(hr_image.shape[1]/self.upscale_factor)*self.upscale_factor
#             y = int(hr_image.shape[2]/self.upscale_factor)*self.upscale_factor
#             hr_image = hr_image[:,:x,:y]
#             self.images.append(hr_image)

#     def __len__(self):
#         return len(self.images)


#     def __getitem__(self, batch_index: int):

#         if self.transforms == None:
#           hr_image = self.images[batch_index]
#         else:
#           hr_image = self.transforms(self.images[batch_index])     
#         return hr_image