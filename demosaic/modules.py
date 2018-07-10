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
import torchlib.viz as viz

from demosaic.dataset import Linear2SRGB

import torchlib.debug as D


def apply_kernels(kernel, noisy_data):
  kh, kw = kernel.shape[2:]
  bs, ci, h, w = noisy_data.shape
  ksize = int(np.sqrt(kernel.shape[1]))

  # Crop kernel and input so their sizes match
  needed = kh + ksize - 1
  if needed > h:
    crop = (needed - h) // 2
    if crop > 0:
      kernel = kernel[:, :, crop:-crop, crop:-crop]
    kh, kw = kernel.shape[2:]
  else:
    crop = (h - needed) // 2
    if crop > 0:
      noisy_data = noisy_data[:, :, crop:-crop, crop:-crop]

  # -------------------------------------------------------------------------
  # Vectorize the kernel tiles
  kernel = kernel.permute(0, 2, 3, 1)
  kernel = kernel.contiguous().view(bs, 1, kh, kw, ksize*ksize)

  # Split the input buffer in tiles matching the kernels
  tiles = noisy_data.unfold(2, ksize, 1).unfold(3, ksize, 1)
  tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)
  # -------------------------------------------------------------------------

  weighted_sum = th.sum(kernel*tiles, dim=4)
  
  return weighted_sum


def get(params):
  params = copy.deepcopy(params)  # do not touch the original
  model_name = params.pop("model", None)
  if model_name is None:
    raise ValueError("model has not been specified!")
  return getattr(sys.modules[__name__], model_name)(**params)

class GreenOnly(nn.Module):
  def __init__(self, mdl):
    super(GreenOnly, self).__init__()
    self.mdl = mdl

  def forward(self, samples):
    output = self.mdl(samples)
    output[:, 0] = 0
    output[:, 2] = 0
    return output

class Subsample(nn.Module):
  def __init__(self, mdl, period, dx=0, dy=0):
    super(Subsample, self).__init__()
    self.mdl = mdl
    self.period = period
    self.dx = dx
    self.dy = dy

  def mask_out(self, arr, start):
    mask = th.zeros_like(arr)
    mask[..., start+self.dy::self.period, start+self.dx::self.period] = 1
    return arr*mask

  def forward(self, samples):
    output = self.mdl(samples)

    oh = output.shape[-2]
    h = samples["target"].shape[-2]

    start = (self.period-((h-oh) //2)) % self.period
    output = self.mask_out(output, start)
    samples["target"] = self.mask_out(samples["target"], 0)

    return output

class DeLinearize(nn.Module):
  def __init__(self, mdl):
    super(DeLinearize, self).__init__()
    self.mdl = mdl
    self.linear2srgb = Linear2SRGB()

  def forward(self, samples):
    output = self.mdl(samples)
    samples["target"] = self.linear2srgb(samples["target"])
    output = self.linear2srgb(output)
    return output


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
    super(XtransNetwork, self).__init__()

    self.depth = depth
    self.width = width

    layers = OrderedDict([])
    for i in range(depth):
      n_in = width
      n_out = width
      if i == 0:
        n_in = 3
      # if i == depth-1:
      #   n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(3+width, width, 3)),
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
  def __init__(self, fov=5, normalize=True):
    super(BayerNN, self).__init__()

    self.fov = fov
    self.normalize = normalize

    self.net = nn.Sequential(
      nn.Linear(fov*fov*4, 128),
      nn.LeakyReLU(inplace=True),
      nn.Linear(128, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, 32),
      nn.LeakyReLU(inplace=True),
      nn.Linear(32, 12),
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
    in_f = in_f.contiguous().view(bs*h*w, c*fov*fov)

    # Log normalize =======
    if self.normalize:
      in_f = th.log(in_f + 1)

      mean_f = in_f.mean(1, keepdim=True)
      in_f -= mean_f

      in_f = th.exp(in_f) - 1.0
    # ---------------------

    in_f = in_f.view(bs*h*w, c*fov*fov)

    out_f = self.net(in_f)

    # Log denormalize =======
    if self.normalize:
      out_f = th.log(th.clamp(out_f, min=-0.5) + 1)  # clip but keep some gradients
      out_f += mean_f
      out_f = th.exp(out_f) - 1.0
    # ---------------------

    out_f = out_f.view(bs, h, w, 3, 2, 2)

    out_f = out_f.permute(0, 3, 1, 4, 2, 5)
    output = out_f.contiguous().view(bs, 3, h*2, w*2)

    return output

class BayerLog(nn.Module):
  """2018-04-01"""
  def __init__(self, fov=7):
    super(BayerLog, self).__init__()

    self.fov = fov

    self.net = nn.Sequential(
      nn.Conv2d(4, 16, 3),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(16, 16, 3),
      nn.LeakyReLU(inplace=True),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(16, 32, 3),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(32, 3, 3),
      )

    self.debug_viz = viz.BatchVisualizer("batch", env="mosaic_debug")
    self.debug_viz2 = viz.BatchVisualizer("batch2", env="mosaic_debug")
    self.debug = False

  def forward(self, samples):
    mosaic = samples["mosaic"]
    gray_mosaic = mosaic.sum(1)

    fov = self.fov
    eps = 1

    color_samples = gray_mosaic.unfold(2, 2, 2).unfold(1, 2, 2)
    color_samples = color_samples.permute(0, 3, 4, 1, 2)
    bs, _, _, h, w = color_samples.shape
    color_samples = color_samples.contiguous().view(bs, 4, h, w)

    in_f = color_samples.unfold(3, fov, 1).unfold(2, fov, 1)
    in_f = in_f.permute(0, 2, 3, 1, 4, 5)
    bs, h, w, c, _, _ = in_f.shape
    in_f = in_f.contiguous().view(bs*h*w, c, fov*fov)

    idx = np.random.randint(0, bs*h*w, size=(128,))
    vdata = in_f.view(bs*h*w, 1, c*fov, fov).cpu().numpy()[idx]

    # Log-normalize
    in_f = th.log(in_f + 1)

    mean_f = in_f.mean(2, keepdim=True)

    mean_r = mean_g = mean_b = (mean_f[:, 0] + mean_f[:, 3])*0.5
    # mean_g = mean_f[:, 0]
    # mean_b = mean_f[:, 0]

    # mean_g = 0.5*(mean_f[:, 0] + mean_f[:, 3])
    # mean_r = mean_f[:, 1]
    # mean_b = mean_f[:, 2]

    in_f[:, 0] -= mean_g
    in_f[:, 1] -= mean_r
    in_f[:, 2] -= mean_b
    in_f[:, 3] -= mean_g

    in_f = th.exp(in_f) - 1.0
    in_f = in_f.view(bs*h*w, c, fov, fov)

    vdata2 = in_f.view(bs*h*w, 1, c*fov, fov).cpu().numpy()[idx]

    if self.debug:
      print(vdata.min().item(), vdata.max().item())
      print(vdata2.min().item(), vdata2.max().item())
      vdata -= vdata.min()
      vdata /= vdata.max()
      vdata2 -= vdata2.min()
      vdata2 /= vdata2.max()
      vdata2 = vdata2.clip(0, 1)
      self.debug_viz.update(vdata,
                            per_row=8, caption="normalized inputs")
      self.debug_viz2.update(vdata2,
                            per_row=8, caption="normalized inputs")

    out_f = self.net(in_f)

    # Log-denormalize
    out_f = th.log(th.clamp(out_f, min=-0.5) + 1)  # clip but keep some gradients
    out_f = out_f.view(bs*h*w, 3, 2*2)
    out_f[:, 0, :] += mean_r
    out_f[:, 1, :] += mean_g
    out_f[:, 2, :] += mean_b
    out_f = th.exp(out_f) - 1.0

    out_f = out_f.view(bs, h, w, 3, 2, 2).permute(0, 3, 1, 4, 2, 5)
    out_f = out_f.contiguous().view(bs, 3, 2*h, 2*w)

    return out_f

class BayerKP(nn.Module):
  """2018-05-18: kernel-predicting Bayer"""
  def __init__(self, ksize=7, convs=2, levels=5, width=64, autoencoder=False):
    """ ksize is the footprint at 1/4 res."""

    super(BayerKP, self).__init__()

    self.ksize = ksize

    if autoencoder:
      self.kernels = Autoencoder(4, 10*ksize*ksize, width=width, increase_factor=1.4,
          num_levels=levels, num_convs=convs, activation="leaky_relu")
          # normalization_type="instance", normalize=True)
    else:
      self.kernels = ConvChain(
          4, 10*ksize*ksize, width=64, depth=levels, pad=False,
          activation="relu", output_type="linear")

  def unroll(self, buf):
    """ Reshapes a 1/4 res buffer with shape [bs, 4, h, w] to fullres.""" 
    bs, _, h, w = buf.shape
    buf = buf.view(bs, 2, 2, h, w).permute(0, 3, 1, 4, 2).contiguous().view(bs, 1, h*2, w*2)
    return buf

  def forward(self, samples, kernels=None):
    start = time.time()

    mosaic = samples["mosaic"]
    gray_mosaic = mosaic.sum(1)

    color_samples = gray_mosaic.unfold(2, 2, 2).unfold(1, 2, 2)
    color_samples = color_samples.permute(0, 3, 4, 1, 2)
    bs, _, _, h, w = color_samples.shape
    color_samples = color_samples.contiguous().view(bs, 4, h, w)

    kernels = self.kernels(color_samples)
    bs, _, h, w = kernels.shape

    # th.cuda.synchronize()
    # elapsed = time.time() - start
    # print("Forward {:.0f} ms".format(elapsed*1000))

    # TODO: check what's going on
    g0 = color_samples[:, 0:1]
    b = color_samples[:, 1:2]
    r = color_samples[:, 2:3]
    g1 = color_samples[:, 3:4]

    idx = 0
    ksize = self.ksize

    # Reconstruct 3 reds from known red
    reds = [r]
    for i in range(3):
      k = kernels[:, idx:idx+ksize*ksize]
      k = F.softmax(k, 1)
      idx += ksize*ksize
      reds.append(apply_kernels(k, r))

    # remove unused boundaries
    reds[0] = crop_like(reds[0], reds[1])

    # Reorder 2x2 tile, known red is 0, pattern is:
    # . R  -> 1 0
    # . .     2 3
    reds = [reds[1], reds[0], reds[2], reds[3]]
    red = self.unroll(th.cat(reds, 1))

    # Reconstruct 3 blues from known blue
    blues = [b]
    for i in range(3):
      k = kernels[:, idx:idx+ksize*ksize]
      k = F.softmax(k, 1)
      idx += ksize*ksize
      blues.append(apply_kernels(k, b))

    # remove unused boundaries
    blues[0] = crop_like(blues[0], blues[1])

    # Reorder 2x2 tile, known blue is 0, pattern is:
    # . .  -> 1 2
    # B .     0 3
    blues = [blues[1], blues[2], blues[0], blues[3]]
    blue = self.unroll(th.cat(blues, 1))

    # Reconstruct 2 greens from known greens
    greens = [g0, g1]
    for i in range(2):
      k = kernels[:, idx:idx + 2*ksize*ksize]
      k = F.softmax(k, 1) # jointly normalize the weights

      from_g0 = apply_kernels(k[:, 0:ksize*ksize], g0)
      from_g1 = apply_kernels(k[:, ksize*ksize:2*ksize*ksize], g1)

      greens.append(from_g0+from_g1)

      idx += 2*ksize*ksize

    # remove unused boundaries
    greens[0] = crop_like(greens[0], greens[2])
    greens[1] = crop_like(greens[1], greens[2])

    # Reorder 2x2 tile, known blue is 0, pattern is:
    # G .  -> 0 2
    # . G     3 1
    greens = [greens[0], greens[2], greens[3], greens[1]]
    green = self.unroll(th.cat(greens, 1))

    output = th.cat([red, green, blue], 1)

    # th.cuda.synchronize()
    # elapsed = time.time() - start
    # print("Forward+Apply {:.0f} ms".format(elapsed*1000))

    return output

class SimpleKP(nn.Module):
  """2018-07-03: kernel-predicting Bayer"""
  def __init__(self, ksize=5, convs=1, levels=3, width=32, mosaic_period=2,
               normalize=False, activation="leaky_relu"):

    super(SimpleKP, self).__init__()

    self.ksize = ksize
    self.normalize = normalize

    self.kernels = Autoencoder(6, 3*ksize*ksize, width=width, increase_factor=1.4,
        num_levels=levels, num_convs=convs, activation=activation,
        normalization_type="instance", normalize=normalize, output_type='linear')

    self.period = mosaic_period

  def forward(self, samples, kernel_list=None):
    start = time.time()

    mosaic = samples["mosaic"]
    mask = samples["mask"]
    gray_mosaic = mosaic.sum(1, keepdim=True)
    bs, _, h, w = gray_mosaic.shape

    x = th.fmod(th.arange(0, w).float().cuda(), self.period).view(1, 1, 1, w).repeat(bs, 1, h, 1)
    y = th.fmod(th.arange(0, h).float().cuda(), self.period).view(1, 1, h, 1).repeat(bs, 1, 1, w)

    indata = th.cat([gray_mosaic, mask, x, y], 1)

    kernels = self.kernels(indata)
    bs, _, h, w = kernels.shape

    idx = 0
    ksize = self.ksize
    chans = []
    for i in range(3):
      k = kernels[:, idx:idx+ksize*ksize]
      k = k / (k.abs().sum(1, keepdim=True) + 1e-4)
      # k = F.softmax(k, 1)
      idx += ksize*ksize
      chans.append(apply_kernels(k, gray_mosaic))

      if kernel_list is not None:
        kernel_list.append(k.view(bs, ksize, ksize, h, w))

    output = th.cat(chans, 1)

    return output

class IndependentKP(nn.Module):
  """
  2018-07-09
  """
  def __init__(self, ksize=3, width=64, period=2):

    super(IndependentKP, self).__init__()

    self.ksize = ksize
    self.period = period
    self.knowns = period*period
    self.unknowns = 2*period*period

    for c in range(self.unknowns):
      net = nn.Sequential(
        nn.Conv2d(self.knowns, width, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(width, width, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(width, ksize*ksize*self.knowns, 1),
      )
      self.add_module(self.get_name(c), net)

  def get_name(self, c):
    return "subnet_{:02d}".format(c)


  def forward(self, samples, kernel_list=None):
    start = time.time()

    mosaic = samples["mosaic"]
    mask = samples["mask"]
    gray_mosaic = mosaic.sum(1, keepdim=True)
    bs, _, h, w = gray_mosaic.shape

    color_samples = gray_mosaic.unfold(
      3, self.period, self.period).unfold(2, self.period, self.period)
    h2, w2 = color_samples.shape[2:4]
    color_samples = color_samples.contiguous().view(
      bs, h2, w2, self.period*self.period).permute(0, 3, 1, 2).contiguous()

    ksize = self.ksize
    chans = []
    for c in range(self.unknowns):
      m = self._modules[self.get_name(c)]
      kernels = m(color_samples)
      bs, _, h, w = kernels.shape
      chans.append(self.apply_kernels(kernels, color_samples))
    recons = th.cat(chans, 1)
    h, w= recons.shape[-2:]

    period = self.period
    recons = recons.view(bs, 2, period, period, h, w)
    recons = recons.permute(0, 1, 4, 2, 5, 3).contiguous().view(bs, 2, h*period, w*period)

    output = th.cat([recons, th.zeros_like(recons)], 1)[:, :3]

    return output

  def apply_kernels(self, kernel, noisy_data):
    kh, kw = kernel.shape[2:]

    bs, ci, h, w = noisy_data.shape
    ksize = self.ksize

    # Crop kernel and input so their sizes match
    needed = kh + ksize - 1
    if needed > h:
      crop = (needed - h) // 2
      if crop > 0:
        kernel = kernel[:, :, crop:-crop, crop:-crop]
      kh, kw = kernel.shape[2:]
    else:
      crop = (h - needed) // 2
      if crop > 0:
        noisy_data = noisy_data[:, :, crop:-crop, crop:-crop]

    # -------------------------------------------------------------------------
    # Split the input buffer in tiles matching the kernels
    tiles = noisy_data.unfold(2, ksize, 1).unfold(3, ksize, 1)
    tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)

    # Vectorize the kernel tiles
    kernel = kernel.view(bs, self.knowns, ksize*ksize, kh, kw)
    kernel = kernel.permute(0, 1, 3, 4, 2).contiguous()
    # kernel = kernel.view(bs, 1, kh, kw, ksize*ksize)
    # -------------------------------------------------------------------------
    weighted_sum = th.sum(th.sum(kernel*tiles, dim=4), dim=1, keepdim=True)

    return weighted_sum


class ClassifierKernels(nn.Module):
  """
  2018-07-10
  """

  def __init__(self, ksize=5, nkernels=8, width=32, period=2):

    super(ClassifierKernels, self).__init__()

    self.ksize = ksize
    self.period = period
    # self.knowns = period*period
    # self.unknowns = 2*period*period
    self.nkernels = nkernels

    self.net = nn.Sequential(
      nn.Conv2d(5, width, 3),
      nn.ReLU(inplace=True),
      nn.Conv2d(width, width, 3),
      nn.ReLU(inplace=True),
      nn.Conv2d(width, nkernels, 1),
    )

    self.kernels = nn.Parameter(th.randn(2, nkernels, ksize*ksize))

  def forward(self, samples, kernel_list=None):
    start = time.time()

    mosaic = samples["mosaic"]
    mask = samples["mask"]
    gray_mosaic = mosaic.sum(1, keepdim=True)
    bs, _, h, w = gray_mosaic.shape

    x = th.fmod(th.arange(0, w).float().cuda(), self.period).view(1, 1, 1, w).repeat(bs, 1, h, 1)
    y = th.fmod(th.arange(0, h).float().cuda(), self.period).view(1, 1, h, 1).repeat(bs, 1, 1, w)

    indata = th.cat([mosaic, x, y], 1)
    weights = self.net(indata)
    weights = F.softmax(weights, 1)

    recons = self.apply_kernels(weights, gray_mosaic.squeeze(1))

    h, w = recons.shape[-2:]

    green = recons.new()
    green.resize_(bs, 1, h, w)
    green.zero_()

    x = crop_like(x, green)
    y = crop_like(y, green)
    gray_mosaic = crop_like(gray_mosaic, green)

    r0 = recons[:, 0:1]
    r1 = recons[:, 1:2]
    
    mask = (x == 1) & (y == 0)
    green[mask] = r0[mask]

    mask = (x == 0) & (y == 1)
    green[mask] = r1[mask]

    green[x == y] = gray_mosaic[x == y] 

    red = th.zeros_like(green)
    blue = th.zeros_like(green)

    output = th.cat([red, green, blue], 1)

    # kview = self.kernels.view(2*self.nkernels, 1, self.ksize, self.ksize)
    # D.tensor(kview)

    return output

  def apply_kernels(self, weights, noisy_data):
    kh, kw = weights.shape[2:]

    bs, h, w = noisy_data.shape
    ksize = self.ksize

    # Crop weights and input so their sizes match
    needed = kh + ksize - 1
    if needed > h:
      crop = (needed - h) // 2
      if crop > 0:
        weights = weights[..., crop:-crop, crop:-crop]
      kh, kw = weights.shape[2:]
    else:
      crop = (h - needed) // 2
      if crop > 0:
        noisy_data = noisy_data[..., crop:-crop, crop:-crop]

    # Split the input buffer in tiles matching the kernels
    tiles = noisy_data.unfold(1, ksize, 1).unfold(2, ksize, 1)
    tiles = tiles.contiguous().view(bs, 1, 1, kh, kw, ksize*ksize)

    kernels = self.kernels.unsqueeze(0).unsqueeze(3).unsqueeze(4)
    weights = weights.unsqueeze(4).unsqueeze(1)

    weighted_kernels = kernels*weights

    # sum over kernel extent
    prod = (tiles*weighted_kernels).sum(2).sum(4)

    return prod
