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

from torchlib.modules import Autoencoder
from torchlib.modules import ConvChain
from torchlib.image import crop_like

def get(params):
  params = copy.deepcopy(params)  # do not touch the original
  model_name = params.pop("model", None)
  if model_name is None:
    raise ValueError("model has not been specified!")
  return getattr(sys.modules[__name__], model_name)(**params)


class BayerNetwork(nn.Module):
  """Released version of the network, best quality.

  This model differs from the published description. It has a mask/filter split
  towards the end of the processing. Masks and filters are multiplied with each
  other. This is not key to performance and can be ignored when training new
  models from scratch.
  """
  def __init__(self, depth=15, width=64):
    super(BayerNetwork, self).__init__()

    self.depth = depth
    self.width = width

    layers = OrderedDict([
        ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance
      ])
    for i in range(depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 4
      if i == depth-1:
        n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)
    self.residual_predictor = nn.Conv2d(width, 12, 1)
    self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(6, width, 3)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    features = self.main_processor(mosaic)
    filters, masks = features[:, :self.width], features[:, self.width:]
    filtered = filters * masks
    residual = self.residual_predictor(filtered)
    upsampled = self.upsampler(residual)

    # crop original mosaic to match output size
    cropped = crop_like(mosaic, upsampled)

    # Concated input samples and residual for further filtering
    packed = th.cat([cropped, upsampled], 1)

    output = self.fullres_processor(packed)

    return output


class XtransNetwork(nn.Module):
  """Released version of the network.

  There is no downsampling here.

  """
  def __init__(self, depth=11, width=64):
    super(BayerNetwork, self).__init__()

    self.depth = depth
    self.width = width

    layers = OrderedDict([])
    for i in range(depth):
      n_out = width
      if i == depth-1:
        n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(width, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(3, width, 3)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    features = self.main_processor(mosaic)

    # crop original mosaic to match output size
    cropped = crop_like(mosaic, features)

    # Concated input samples and residual for further filtering
    packed = th.cat([cropped, features], 1)

    output = self.fullres_processor(packed)

    return output


class L2Loss(nn.Module):
  """ """
  def __init__(self):
    super(L2Loss, self).__init__()
    self.mse = nn.MSELoss()

  def forward(self, data, output):
    target = crop_like(data["target"], output)
    return self.mse(output, target)


class PSNR(nn.Module):
  """ """
  def __init__(self):
    super(PSNR, self).__init__()
    self.mse = nn.MSELoss()

  def forward(self, data, output):
    target = crop_like(data["target"], output)
    mse = self.mse(output, target)
    return -10 * th.log(mse) / np.log(10)
