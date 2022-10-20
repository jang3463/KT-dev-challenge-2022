import torch.nn as nn
import numpy as np

class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
  
  def forward(self, x):
    return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=3, patch_size=3, n_classes=50):
  return nn.Sequential(
      nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
      nn.GELU(),
      nn.BatchNorm2d(dim),
      *[nn.Sequential(
          Residual(nn.Sequential(
              nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
              nn.GELU(),
              nn.BatchNorm2d(dim)
          )),
          nn.Conv2d(dim, dim, kernel_size=1),
          nn.GELU(),
          nn.BatchNorm2d(dim)
      ) for i in range(depth)],
      nn.AdaptiveAvgPool2d((1,1)),
      nn.Flatten(),
      nn.Linear(dim, n_classes)
  )


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        n = model.in_features
        nn.init.normal_(model.weight.data, 0.0, np.sqrt(2 / n))
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)