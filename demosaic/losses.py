import os
import sys
import copy
import time
from collections import OrderedDict

import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

from torchlib.modules import Autoencoder
from torchlib.modules import ConvChain
from torchlib.image import crop_like
import torchlib.viz as viz

class L2Loss(nn.Module):
  """ """
  def __init__(self, weight=1.0):
    super(L2Loss, self).__init__()
    self.mse = nn.MSELoss()
    self.weight = weight

  def forward(self, data, output):
    target = crop_like(data["target"], output)
    return self.mse(output, target) * self.weight


class PSNR(nn.Module):
  """ """
  def __init__(self):
    super(PSNR, self).__init__()
    self.mse = nn.MSELoss()

  def forward(self, data, output):
    target = crop_like(data["target"], output)
    mse = self.mse(output, target) + 1e-12
    return -10 * th.log(mse) / np.log(10)


class VGGLoss(nn.Module):
  """ """
  def __init__(self, weight=1.0, normalize=True):
    super(VGGLoss, self).__init__()

    self.normalize = normalize
    self.weight = weight

    vgg = tmodels.vgg16(pretrained=True).features
    slices_idx = [
        [0, 4],
        [4, 9],
        [9, 16],
        [16, 23],
        [23, 30],
        ]
    self.net = th.nn.Sequential()
    for i, idx in enumerate(slices_idx):
      seq = th.nn.Sequential()
      for j in range(idx[0], idx[1]):
        seq.add_module(str(j), vgg[j])
      self.net.add_module(str(i), seq)

    for p in self.parameters():
      p.requires_grad = False

    self.mse = nn.MSELoss()

  def forward(self, data, output):
    target = crop_like(data["target"], output)
    output_f = self.get_features(output)
    with th.no_grad():
      target_f = self.get_features(target)

    losses = []
    for o, t in zip(output_f, target_f):
      losses.append(self.mse(o, t))
    loss = sum(losses)
    if self.weight != 1.0:
      loss = loss * self.weight
    return loss 

  def get_features(self, x):
    """Assumes x in [0, 1]: transform to [-1, 1]."""
    x = 2.0*x - 1.0
    feats = []
    for i, s in enumerate(self.net):
      x = s(x)
      if self.normalize:  # unit L2 norm over features, this implies the loss is a cosine loss in feature space 
        f = x / (th.sqrt(th.pow(x, 2).sum(1, keepdim=True)) + 1e-8)
      else:
        f = x
      feats.append(f)
    return feats
