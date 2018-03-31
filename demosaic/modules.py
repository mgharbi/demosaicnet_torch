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
import torchvision.models as tvmodels

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


class BayerExperimental(nn.Module):
  """2018-03-30"""
  def __init__(self, depth=4, width=32):
    super(BayerExperimental, self).__init__()

    self.depth = depth
    self.width = width

    self.fov = (depth-1)*2 + 1

    # self.averager = nn.AvgPool2d(fov, padding=(fov-1)/2, count_include_pad=False)

    self.local_mean = nn.Conv2d(4, 1, 3, bias=False, padding=1)
    self.local_mean.weight.data.fill_(1.0/(9.0*4))

    layers = OrderedDict()
    for i in range(depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 4
      if i < depth - 1:
        layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
        layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)
      else:
        layers["output"] = nn.Conv2d(n_in, 8, 1)
    self.net = nn.Sequential(layers)

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    gray_mosaic = mosaic.sum(1)
    color_samples = gray_mosaic.unfold(2, 2, 2).unfold(1, 2, 2)
    color_samples = color_samples.permute(0, 3, 4, 1, 2)
    bs, _, _, h, w = color_samples.shape
    color_samples = color_samples.contiguous().view(bs, 4, h, w)

    eps = 1e-8

    color_samples = th.log(color_samples + eps)

    # input_mean = color_samples.mean(1, keepdim=True)
    input_mean = self.local_mean(color_samples)

    # recons_samples = self.net(color_samples)
    recons_samples = self.net(color_samples-input_mean)

    cmean = crop_like(input_mean, recons_samples)

    recons_samples = recons_samples + cmean

    recons_samples = th.exp(recons_samples) - 1e-8

    _, _, h, w = recons_samples.shape

    output = mosaic.new()
    output.resize_(bs, 3, 2*h, 2*w) 
    output.zero_()

    cmosaic = crop_like(mosaic, output)

    # has green
    output[:, 0, ::2, ::2] = recons_samples[:, 0]
    output[:, 1, ::2, ::2] = cmosaic[:, 1, ::2, ::2]
    output[:, 2, ::2, ::2] = recons_samples[:, 1]

    # has red
    output[:, 0, ::2, 1::2] = cmosaic[:, 0, ::2, 1::2]
    output[:, 1, ::2, 1::2] = recons_samples[:, 2]
    output[:, 2, ::2, 1::2] = recons_samples[:, 3]

    # has blue
    output[:, 0, 1::2, 0::2] = recons_samples[:, 4]
    output[:, 1, 1::2, 0::2] = recons_samples[:, 5]
    output[:, 2, 1::2, 0::2] = cmosaic[:, 2, 1::2, 0::2]

    # has green
    output[:, 0, 1::2, 1::2] = recons_samples[:, 6]
    output[:, 1, 1::2, 1::2] = cmosaic[:, 1, 1::2, 1::2]
    output[:, 2, 1::2, 1::2] = recons_samples[:, 7]

    return output

class BayerNN(nn.Module):
  """2018-03-30"""
  def __init__(self, fov=5):
    super(BayerNN, self).__init__()

    self.fov = fov

    self.net = nn.Sequential(
      nn.Linear(fov*fov*4, 128),
      nn.LeakyReLU(inplace=True),
      nn.Linear(128, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, 32),
      nn.LeakyReLU(inplace=True),
      nn.Linear(32, 8),
      )


  def forward(self, samples):
    mosaic = samples["mosaic"]
    gray_mosaic = mosaic.sum(1)

    fov = self.fov
    eps = 1e-8

    color_samples = gray_mosaic.unfold(2, 2, 2).unfold(1, 2, 2)
    color_samples = color_samples.permute(0, 3, 4, 1, 2)
    bs, _, _, h, w = color_samples.shape
    color_samples = color_samples.contiguous().view(bs, 4, h, w)

    in_f = color_samples.unfold(3, fov, 1).unfold(2, fov, 1)
    in_f = in_f.permute(0, 2, 3, 1, 4, 5)
    bs, h, w, c, _, _ = in_f.shape
    in_f = in_f.contiguous().view(bs, h*w, c*fov*fov)

    mean_f = in_f.mean(2, keepdim=True)
    nrm = mean_f + eps

    in_f /= nrm

    out_f = self.net(in_f)
    out_f = out_f.view(bs, h*w, 8)
    out_f *= nrm

    out_f = out_f.view(bs, h, w, 8).permute(0, 3, 1, 2)

    output = mosaic.new()
    output.resize_(bs, 3, 2*h, 2*w) 
    output.zero_()

    cmosaic = crop_like(mosaic, output)

    # has green
    output[:, 0, ::2, ::2] = out_f[:, 0]
    output[:, 1, ::2, ::2] = cmosaic[:, 1, ::2, ::2]
    output[:, 2, ::2, ::2] = out_f[:, 1]

    # has red
    output[:, 0, ::2, 1::2] = cmosaic[:, 0, ::2, 1::2]
    output[:, 1, ::2, 1::2] = out_f[:, 2]
    output[:, 2, ::2, 1::2] = out_f[:, 3]

    # has blue
    output[:, 0, 1::2, 0::2] = out_f[:, 4]
    output[:, 1, 1::2, 0::2] = out_f[:, 5]
    output[:, 2, 1::2, 0::2] = cmosaic[:, 2, 1::2, 0::2]

    # has green
    output[:, 0, 1::2, 1::2] = out_f[:, 6]
    output[:, 1, 1::2, 1::2] = cmosaic[:, 1, 1::2, 1::2]
    output[:, 2, 1::2, 1::2] = out_f[:, 7]

    return output


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
    mse = self.mse(output, target)
    return -10 * th.log(mse) / np.log(10)


class VGGLoss(nn.Module):
  """ """
  def __init__(self, weight=1.0):
    super(VGGLoss, self).__init__()
    self.mse = nn.MSELoss()
    self.weight = weight

    self.vgg = tvmodels.vgg19(pretrained=True)
    self.layers = self.vgg.features
    self.layer_name_mapping = {
        '3': "relu1_2",
        '8': "relu2_2",
        '15': "relu3_3",
        '22': "relu4_3"
        }

  def forward(self, data, output):
    target = crop_like(data["target"], output)
    output_f = self.get_features(output)
    with th.no_grad():
      target_f = self.get_features(target)
    losses = []
    for k in output_f:
      losses.append(self.mse(output_f[k], target_f[k]))
    return sum(losses) * self.weight

  def get_features(self, x):
    output = {}
    for name, module in self.layers._modules.items():
      x = module(x)
      if name in self.layer_name_mapping:
        output[self.layer_name_mapping[name]] = x
    return output
