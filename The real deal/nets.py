import torch
import numpy as np
import math

import torch
from torch import nn
from torch import optim


class SRCNN(nn.Module):
    def __init__(self,nBaseChannels,upSamplingFactor) -> None:
        super(SRCNN, self).__init__()

        self.features = nn.Sequential(
        nn.Upsample(scale_factor=upSamplingFactor),  
        nn.Conv2d(3, nBaseChannels, 9,1,4),
        nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
        nn.Conv2d(nBaseChannels, nBaseChannels, 5,1,2),
        nn.ReLU(True),
        )

        # Reconstruction layer.
        self.reconstruction = nn.Conv2d(nBaseChannels, 3, 5,1,2)

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return torch.clip(out,min=0,max=1)

# The filter weight of each layer is a Gaussian distribution with zero mean and
# standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
      for module in self.modules():
          if isinstance(module, nn.Conv2d):
              torch.nn.init.xavier_uniform_(module.weight)
              #nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
              nn.init.zeros_(module.bias.data)

      nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
      nn.init.zeros_(self.reconstruction.bias.data)
        
        
from torch.nn.modules.activation import ReLU
class ResidualBlock(nn.Module):
    
    # YOUR CODE HERE
    def __init__(self, in_channels, out_channels, kernel_size):
      super(ResidualBlock, self).__init__();

      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size

      self.block = nn.Sequential(
          nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, 2),
          nn.ReLU(True),
          nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, 2)
      )

    def forward(self, x: torch.tensor) -> torch.tensor:
      x =  x + self.block(x)
      x = torch.nn.functional.relu(x)
      return x

# class SRResnet(nn.Module):

#     def __init__(self, nBaseChannels, upSamplingFactor)-> None:
#         super(SRResnet, self).__init__();

#         # Feature extraction layer.
#         self.features = nn.Sequential(
#         nn.Upsample(scale_factor=upSamplingFactor),  
#         nn.Conv2d(3, nBaseChannels, 9,1,4),
#         nn.ReLU(True)
#         )

#         # Non-linear mapping layer.
#         self.map = nn.Sequential(
#           ResidualBlock(nBaseChannels, nBaseChannels, 5),
#           ResidualBlock(nBaseChannels, nBaseChannels, 5),
#           ResidualBlock(nBaseChannels, nBaseChannels, 5),
#           ResidualBlock(nBaseChannels, nBaseChannels, 5),
#           ResidualBlock(nBaseChannels, nBaseChannels, 5)
#         )

#         # Reconstruction layer.
#         self.reconstruction = nn.Conv2d(nBaseChannels, 3, 5,1,2)

#         # Initialize model weights.
#         self._initialize_weights()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.features(x)
#         out = self.map(out)
#         out = self.reconstruction(out)
#         return torch.clip(out,min=0,max=1)

#     # The filter weight of each layer is a Gaussian distribution with zero mean and
#     # standard deviation initialized by random extraction 0.001 (deviation is 0)
#     def _initialize_weights(self) -> None:
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d):
#               torch.nn.init.xavier_normal_(module.weight)
#               nn.init.zeros_(module.bias.data)

#         nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
#         nn.init.zeros_(self.reconstruction.bias.data)