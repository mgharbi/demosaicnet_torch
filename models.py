import os
import sys
import copy
import time

import numpy as np
from torch.autograd import Variable
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import rendernet.modules.preprocessors as pre
import rendernet.modules.operators as ops
import rendernet.utils as rutils

from torchlib.image import crop_like
from torchlib.modules import ConvChain
from torchlib.modules import Autoencoder
from torchlib.modules import RecurrentAutoencoder
from torchlib.modules import FullyRecurrentAutoencoder
from torchlib.modules import FullyConnected
from torchlib.modules import RecurrentCNN
import torchlib.viz as viz

from torchlib.utils import Timer

import rendernet.viz as rviz

import skimage.io as skio


def get(preprocessor, params, requires_grad=True):
  params = copy.deepcopy(params)  # do not touch the original
  model_name = params.pop("model", None)
  if model_name is None:
    raise ValueError("model has not been specified!")
  mdl = getattr(sys.modules[__name__], model_name)(preprocessor, **params)
  if not requires_grad:
    for p in mdl.parameters():
      p.requires_grad = False
  return mdl

def apply_kernels(kernel, noisy_data, normalization="sum"):
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
  
  if normalization == "sum":
    kernel_sum = th.sum(kernel, dim=4)
  elif normalization == "l1":
    kernel_sum = th.sum(th.abs(kernel), dim=4)
  elif normalization == "none":
    kernel_sum = None

  return weighted_sum, kernel_sum

class DirectAE(nn.Module):
  """
  Direct autoencoder baseline
  2018-02-22
  """
  def __init__(self, pre, ksize=21, width=128, num_levels=3, bayesian=False,
               randomize_spp=False):
    super(DirectAE, self).__init__()

    assert ksize >= 3, "ksize should be greater than 3"
    assert ksize % 2 == 1, "ksize should be odd"

    self.pre = pre
    self.num_levels = num_levels
    self.ksize = ksize
    self.bayesian = bayesian

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures
    self.main_processor = Autoencoder(
        nfeats+3, 3, num_levels=self.num_levels, 
        increase_factor=2.0, 
        num_convs=3, width=width, ksize=3,
        output_type="leaky_relu", pooling="max")

    self.randomize_spp = randomize_spp


  def forward(self, samples):
    noisy_data = samples["radiance"]
    features = samples["features"]

    bs, spp, ci, h, w = features.shape

    if self.randomize_spp and self.training:
      s = np.random.randint(min(2, spp), spp+1)
      noisy_data = noisy_data[:, :s, ...]
      features = features[:, :s, ...]

    noisy_data = noisy_data.mean(1)
    features = features.mean(1)

    gfeatures = samples["global_features"]
    if gfeatures is not None:
      gfeatures = gfeatures.repeat([1, 1, h, w])

    input_map = th.cat([features, gfeatures, noisy_data], 1)
    output = self.main_processor(input_map)

    return output, None, None, None


class KPAE(nn.Module):
  """
  Kernel-predicting autoencoder baseline
  2018-02-22
  """
  def __init__(self, pre, ksize=21, width=128, num_levels=3, bayesian=False,
               randomize_spp=False, autoencoder=True, softmax=True):
    super(KPAE, self).__init__()

    assert ksize >= 3, "ksize should be greater than 3"
    assert ksize % 2 == 1, "ksize should be odd"

    self.pre = pre
    self.num_levels = num_levels
    self.ksize = ksize
    self.bayesian = bayesian

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures

    self.debug = rutils.DebugVisualizer(
        ["features", "kernel_diffuse", "kernel_specular", "prekernel", 
          "prekernel_g", "mix"], 
        env="kpae_albedo", debug=True, period=1)

    if autoencoder:
      self.main_processor = Autoencoder(
          nfeats,
          width, num_levels=self.num_levels, 
          increase_factor=2.0, 
          num_convs=3, width=width, ksize=3,
          output_type="leaky_relu", pooling="max")
    else:
      self.main_processor = ConvChain(
          nfeats, width, depth=12,
          activation="leaky_relu", output_type="linear",
          pad=True)

    self.kernel_regressor = ConvChain(
        width + nfeats, ksize*ksize, depth=3, 
        width=width, ksize=3,
        activation="leaky_relu",
        pad=False, output_type="linear")

    if self.bayesian:
      self.variance_regressor = ConvChain(
          width, 1, depth=3, 
          width=width, ksize=3, normalize=False,
          activation="leaky_relu",
          pad=True, output_type="linear")

    self.randomize_spp = randomize_spp

  def forward(self, samples, kernel_list=None):
    radiance = samples["radiance"]
    features = samples["features"]

    bs, spp, ci, h, w = features.shape

    if self.randomize_spp and self.training:
      s = np.random.randint(min(2, spp), spp+1)
      radiance = radiance[:, :s, ...]
      features = features[:, :s, ...]

    radiance = radiance.mean(1)
    features = features.mean(1)

    gfeatures = samples["global_features"]
    if gfeatures is not None:
      gfeatures = gfeatures.repeat([1, 1, h, w])

    input_map = th.cat([features, gfeatures], 1)
    self.debug.step()

    ae_output = self.main_processor(input_map)

    # Skip connection
    kernel = self.kernel_regressor(th.cat([ae_output, input_map], 1))
    kernel = F.softmax(kernel, dim=1)  # KP-style normalization

    if kernel_list is not None:
      kernel_list.append(kernel)

    output, _ = apply_kernels(kernel, radiance, normalization="l1")

    output = {"radiance": output}

    log_variance = None
    if self.bayesian:  # Predict variance if needed
      feats = crop_like(ae_output, output)
      log_variance = self.variance_regressor(feats)
      output["variance"] = log_variance

    return output


class KPRNN(nn.Module):
  """
  2018-01-18
  """
  def __init__(self, ninputs, width=128, ksize=21, num_levels=3,
               normalize=False, pad=True, kernel_type="softplus", autoencoder=True):
    super(KPRNN, self).__init__()

    assert ksize >= 3, "ksize should be greater than 3"
    assert ksize % 2 == 1, "ksize should be odd"

    self.ninputs = ninputs
    self.num_levels = num_levels
    self.ksize = ksize
    self.kernel_type = kernel_type

    keys = ["rnn", "rnn_g", "kernel", "kernel_g", "mixing"]
    self.debug = rutils.DebugVisualizer(keys, env="debug", debug=True)
    self.imdebug = rutils.DebugFeatureVisualizer(
        ["residual"], env="debug", debug=True, period=10, per_chan=False)

    self.eps = 1e-8

    if autoencoder:
      self.main_processor = RecurrentAutoencoder(
          self.ninputs,
          width, num_levels=num_levels, increase_factor=2.0, 
          num_convs_state=2,
          num_convs=2, width=width, 
          ksize=3, temporal_ksize=3,
          activation="leaky_relu",
          recurrent_activation="leaky_relu",
          output_type="leaky_relu", pooling="max")
    else:
      self.main_processor = RecurrentCNN(
          self.ninputs, width, width=width, block_depth=2, nblocks=5)

    # TODO: soft-nonnegative kernels
    # out_type = "relu"
    out_type = "linear"
    k_out = ksize*ksize
    if "sigmoid" in self.kernel_type:
      out_type = "sigmoid"
    elif "relu" in self.kernel_type:
      out_type = "relu"
    elif "softplus" in self.kernel_type:
      out_type = "softplus"
    self.kernel_regressor = ConvChain(
        width + self.ninputs, k_out, depth=3, 
        width=width, ksize=3, 
        activation="leaky_relu",
        pad=False, output_type=out_type)

  def predict_and_apply_kernels(
      self, rnn_features, features, radiance, irradiance, albedo, kernel_list):

    # Skip connection
    nf = th.cat([rnn_features, features], 1)

    kernel = self.kernel_regressor(nf)

    self.debug.register_grad("kernel_g", kernel)

    pre_w = None
    nk = 1

    # Weight local samples and integrate
    if "l1" in self.kernel_type :
      weighted_sum, kernel_sum = apply_kernels(kernel, radiance, normalization="l1")
    else:
      weighted_sum, kernel_sum = apply_kernels(kernel, radiance, normalization="sum")

    if kernel_list is not None:
      kernel_list.append(kernel)

    return weighted_sum, kernel_sum, pre_w

  def forward(self, features, radiance, irradiance, albedo, gfeatures=None, kernel_list=None):
    bs, spp, ci, h, w = radiance.shape
    ksize = self.ksize

    irr = None
    alb = None

    if gfeatures is not None:
      gfeatures = gfeatures.repeat([1, 1, h, w])

    # # Warmup steps
    # for step in range(min(spp, 4)):
    #   f = self.get_features(step, features, gfeatures)
    #   if step == 0:
    #     # Zero state
    #     state = self.main_processor.get_init_state(features[:, 0])
    #
    #   rnn_features, state = self.main_processor(f, state, encoder_only=True)

    # Actual RNN evaluation
    for step in range(spp):
      f = self.get_features(step, features, gfeatures)

      if step == 0:
        # Zero state
        state = self.main_processor.get_init_state(features[:, 0])

      if irradiance is not None:
        irr = irradiance[:, step]

      # TODO
      # if albedo is not None:
      #   alb = albedo[:, step]
      alb = None

      # Advance state 
      rnn_features, state = self.main_processor(f, state)

      # # Predict kernels
      # weighted_sum, kernel_sum, pre_w = self.predict_and_apply_kernels(
      #     rnn_features, f, radiance[:, step], irr, alb, kernel_list)

      # Skip connection
      nf = th.cat([rnn_features, f], 1)

      kernel = self.kernel_regressor(nf)

      if self.kernel_type == "softmax":
        if step == 0:
          max_kernel = kernel.view(bs, -1).max(1)[0].view(bs, 1, 1, 1)
        else:
          old_max = max_kernel
          max_kernel = th.max(old_max, kernel.view(bs, -1).max(1)[0].view(bs, 1, 1, 1))

        # well behaved softmax
        kernel = th.exp(kernel - max_kernel)

      # Weight local samples and integrate
      if "l1" in self.kernel_type :
        weighted_sum, kernel_sum = apply_kernels(kernel, radiance[:, step], normalization="l1")
      else:
        weighted_sum, kernel_sum = apply_kernels(kernel, radiance[:, step], normalization="sum")

      if kernel_list is not None:
        kernel_list.append(kernel)


      if step == 0:
        # Initialize output buffers
        unnormalized = weighted_sum
        sum_weights = kernel_sum
      else:
        # Update unnormalized output, and normalizer
        if self.kernel_type == "softmax":
          unnormalized = unnormalized*th.exp(old_max - max_kernel) + weighted_sum
          sum_weights = sum_weights*th.exp(old_max - max_kernel) + kernel_sum
        else:
          unnormalized = unnormalized + weighted_sum
          sum_weights = sum_weights + kernel_sum

    self.debug.step()
    self.debug.update("rnn", rnn_features)
    self.debug.register_grad("rnn_g", rnn_features)

    # Normalize output
    output = unnormalized / (sum_weights + self.eps)

    return {
        "radiance": output,
        "rnn_features": rnn_features,
        # "sum_w": sum_weights,
    }

  def get_features(self, step, features, gfeatures):
    if gfeatures is not None:
      f = [features[:, step, ...]]
      if gfeatures is not None:
        f.append(gfeatures)
      f = th.cat(f, 1)
    else:
      f = features[:, step, ...]
    return f

class DirectRNN(nn.Module):
  """
  2018-04-14
  """
  def __init__(self, ninputs, width=128, num_levels=3, autoencoder=True):
    super(DirectRNN, self).__init__()

    self.ninputs = ninputs
    self.num_levels = num_levels

    if autoencoder:
      self.main_processor = RecurrentAutoencoder(
          self.ninputs+3,
          width, num_levels=num_levels, increase_factor=2.0, 
          num_convs_state=2,
          num_convs=2, width=width, 
          ksize=3, temporal_ksize=3,
          activation="leaky_relu",
          recurrent_activation="leaky_relu",
          output_type="leaky_relu", pooling="max")
    else:
      self.main_processor = RecurrentCNN(
          self.ninputs, width, width=width, block_depth=2, nblocks=5)

    self.output_regressor = ConvChain(
        width, 3, depth=1, 
        width=width, ksize=1, 
        activation="leaky_relu",
        pad=False, output_type="linear")

    self.debug = rutils.DebugVisualizer(
        ["rnn_output", "rnn_output_g", "output"], env="debug", debug=True)

  def forward(self, features, radiance, irradiance, 
              albedo, gfeatures=None, kernel_list=None):
    bs, spp, ci, h, w = radiance.shape

    if gfeatures is not None:
      gfeatures = gfeatures.repeat([1, 1, h, w])

    # Actual RNN evaluation
    for step in range(spp):
      f = self.get_features(step, features, gfeatures)

      f = th.cat([f, radiance[:, step]], 1)

      if step == 0:
        # Zero state
        state = self.main_processor.get_init_state(f)

      # Advance state
      if step == spp-1:
        rnn_out, state = self.main_processor(f, state)
        output = self.output_regressor(rnn_out)
        self.debug.step()
        self.debug.update("rnn_output", rnn_out)
        self.debug.register_grad("rnn_output_g", rnn_out)
        self.debug.update("output", output)
      else:
        _, state = self.main_processor(f, state)

    return {
        "radiance": output,
    }

  def get_features(self, step, features, gfeatures):
    if gfeatures is not None:
      f = [features[:, step, ...]]
      if gfeatures is not None:
        f.append(gfeatures)
      f = th.cat(f, 1)
    else:
      f = features[:, step, ...]
    return f

class AlbedoKPRNN(nn.Module):
  """
  2018-01-18
  """
  def __init__(self, ninputs, width=128, ksize=21, num_levels=3,
               normalize=False, pad=True,
               kernel_type="l1", fully_recurrent=False):
    super(AlbedoKPRNN, self).__init__()

    assert ksize >= 3, "ksize should be greater than 3"
    assert ksize % 2 == 1, "ksize should be odd"

    self.ninputs = ninputs
    self.num_levels = num_levels
    self.ksize = ksize
    self.kernel_type = kernel_type

    keys = ["rnn", "kernel_i", "kernel", "mix"]
    self.debug = rutils.DebugVisualizer(keys, env="debug", debug=True)

    self.eps = 1e-8

    self.main_processor = FullyRecurrentAutoencoder(
        self.ninputs,
        width, num_levels=num_levels, increase_factor=1.3, 
        num_convs_state=2,
        num_convs=2, width=width, 
        ksize=3, temporal_ksize=3,
        activation="leaky_relu",
        recurrent_activation="leaky_relu",
        output_type="leaky_relu", pooling="max")

    # TODO: soft-nonnegative kernels
    self.kernel_regressor = ConvChain(
        width + self.ninputs, 
        ksize*ksize*2, depth=3, 
        width=width, ksize=3, 
        activation="leaky_relu",
        pad=False, output_type="softplus")

    self.mix_regressor = ConvChain(
        width, 
        1, depth=3, 
        width=width, ksize=3, 
        activation="leaky_relu",
        pad=False, output_type="sigmoid")

  def predict_and_apply_kernels(
      self, rnn_features, features, radiance, irradiance, albedo, kernel_list):

    nf = th.cat([rnn_features, features], 1)
    kernel = self.kernel_regressor(nf)

    # split radiance/irradiance kernels
    ksize = self.ksize
    kernel_r = kernel[:, ksize*ksize:]
    kernel = kernel[:, :ksize*ksize]

    self.debug.update("kernel", kernel_r)
    self.debug.update("kernel_i", kernel)

    # Reconstruct irradiance
    wsum, ksum = apply_kernels(kernel, irradiance, normalization="l1")
    # Reconstruct radiance
    wsum_r, ksum_r = apply_kernels(kernel_r, radiance, normalization="l1")

    return [wsum, wsum_r], [ksum, ksum_r]

  def forward(
      self, features, radiance, irradiance, albedo,
      gfeatures=None, kernel_list=None):

    bs, spp, ci, h, w = radiance.shape
    ksize = self.ksize

    if gfeatures is not None:
      gfeatures = gfeatures.repeat([1, 1, h, w])

    # Actual RNN evaluation
    for step in range(spp):
      f = self.get_features(step, features, gfeatures)

      if step == 0:
        # Zero state
        state = self.main_processor.get_init_state(features[:, 0])

      # Advance state 
      rnn_features, state = self.main_processor(f, state)

      # Predict kernels
      weighted_sum, kernel_sum = self.predict_and_apply_kernels(
          rnn_features, f, radiance[:, step], irradiance[:, step], 
          albedo[:, step], kernel_list)

      if step == 0:
        # Initialize output buffers
        output = weighted_sum
        sum_weights = kernel_sum
      else:
        # Update unnormalized output, and normalizer
        for i in range(2):
          output[i] = output[i] + weighted_sum[i]
          sum_weights[i] = sum_weights[i] + kernel_sum[i]

    self.debug.step()

    # mix = self.mix_regressor(rnn_features)
    # mix = crop_like(mix, output[0])

    self.debug.update("rnn", rnn_features)
    # self.debug.update("mix", mix)

    # Normalize output
    recons_albedo = crop_like(albedo.mean(1), weighted_sum[0])
    # recons_irradiance = output[0] / (sum_weights[0] + self.eps)
    # recons_radiance = output[1] / (sum_weights[1] + self.eps)

    # recons_radiance = (output[0] * recons_albedo) / (sum_weights[0] + sum_weights[1] + self.eps)
    recons_radiance = (output[0] * recons_albedo + output[1]) / (sum_weights[0] + sum_weights[1] + self.eps)

    # recons_radiance = (recons_irradiance * recons_albedo)*mix + recons_radiance*(1-mix)
    # recons_radiance = (recons_irradiance * recons_albedo) + recons_radiance

    return {
        "radiance": recons_radiance,
        # "irradiance": recons_irradiance,
        "albedo": recons_albedo,
        "rnn_features": rnn_features,
        "sum_w": sum_weights[0] + sum_weights[1],
    }


  def get_features(self, step, features, gfeatures):
    if gfeatures is not None:
      f = [features[:, step, ...]]
      if gfeatures is not None:
        f.append(gfeatures)
      f = th.cat(f, 1)
    else:
      f = features[:, step, ...]
    return f


class FixedRenderer(nn.Module):
  def __init__(self, pre, width=128, bayesian=False, ksize=21,
               num_levels=3,
               predictor="kprnn",
               randomize_spp=True, kernel_type="l1", add_residual=False, autoencoder=True):
    super(FixedRenderer, self).__init__()
    self.pre = pre
    self.predictor = predictor

    if predictor == "kprnn":
      self.predictor = KPRNN(self.pre.nfeatures + self.pre.n_gfeatures, width=width,
                         num_levels=num_levels, ksize=ksize, kernel_type=kernel_type,
                         autoencoder=autoencoder)
    elif predictor == "akprnn":
      self.predictor = AlbedoKPRNN(self.pre.nfeatures + self.pre.n_gfeatures, width=width,
                         num_levels=num_levels, ksize=ksize, kernel_type=kernel_type)
    elif predictor == "rnn":
      self.predictor = DirectRNN(self.pre.nfeatures + self.pre.n_gfeatures, width=width,
                                 num_levels=num_levels)
    else:
      raise ValueError("unrecognized predictor")

    self.randomize_spp = randomize_spp

    self.bayesian = bayesian
    if self.bayesian:
      self.variance_regressor = ConvChain(
          width, 1, depth=6, 
          width=width, ksize=3, normalize=False,
          activation="leaky_relu",
          pad=False, output_type="linear")

  def forward(self, samples, kernel_list=None):
    radiance = samples["radiance"]
    irradiance = None
    albedo = None
    if "irradiance" in samples.keys():
      irradiance = samples["irradiance"]
    if "albedo" in samples.keys():
      albedo = samples["albedo"]
    features = samples["features"]
    gfeatures = samples["global_features"]

    s = None
    bs, spp = radiance.shape[:2]
    if self.randomize_spp and self.training:
      s = np.random.randint(1, spp+1)
      radiance = radiance[:, :s, ...]
      if irradiance is not None:
        irradiance = irradiance[:, :s, ...]
      if albedo is not None:
        albedo = albedo[:, :s, ...]
      features = features[:, :s, ...]

    output = self.predictor(
        features, radiance, irradiance, albedo, gfeatures, 
        kernel_list=kernel_list)

    # Predict variance if needed
    if self.bayesian:
      feats = crop_like(output["rnn_features"], output["radiance"])

      log_variance = self.variance_regressor(feats)

      log_variance = th.clamp(log_variance, -10, 10)  # prevents diverging loss?

      log_variance = crop_like(log_variance, output["radiance"])
      output["variance"] = log_variance

    return output


class AdaptiveRenderer(nn.Module):
  """
  2018-04-15
  """
  def __init__(self, pre, width=128, ksize=21, num_levels=3,
               normalize=False, pad=True, bayesian=False, ratio=0.1,
               randomize_spp=True):
    super(AdaptiveRenderer, self).__init__()

    self.pre = pre

    assert ksize >= 3, "ksize should be greater than 3"
    assert ksize % 2 == 1, "ksize should be odd"
    assert bayesian == True

    self.ninputs = self.pre.nfeatures + self.pre.n_gfeatures
    self.num_levels = num_levels
    self.ksize = ksize
    self.ratio = float(ratio)  # fraction of new samples
    self.randomize_spp = randomize_spp

    self.eps = 1e-8

    self.main_processor = RecurrentAutoencoder(
        self.ninputs + 1,
        width, num_levels=num_levels, increase_factor=2.0, 
        num_convs_state=2,
        num_convs=2, width=width, 
        ksize=3, temporal_ksize=3,
        activation="leaky_relu",
        recurrent_activation="leaky_relu",
        output_type="leaky_relu", pooling="max")

    self.kernel_regressor = ConvChain(
        width + self.ninputs + 1, ksize*ksize, depth=3, 
        width=width, ksize=3, 
        normalize=False, normalization_type="instance",
        activation="leaky_relu",
        pad=True, output_type="softplus")

    self.variance_regressor = ConvChain(
        width + 1, 1, depth=6, 
        width=width, ksize=3, normalize=False,
        activation="leaky_relu",
        pad=True, output_type="linear")

    keys = ["mask_{}".format(i) for i in range(8)]
    self.imdebug = rutils.DebugFeatureVisualizer(
        keys, env="adaptive", debug=True, period=10)
    self.debug = rutils.DebugVisualizer(
        ["kernel"], env="adaptive", debug=True)

  def predict_and_apply_kernels(
      self, rnn_features, features, radiance, mask, kernel_list):

    # Skip connection
    nf = th.cat([rnn_features, features, mask], 1)

    kernel = self.kernel_regressor(nf)
    self.debug.step()
    self.debug.update("kernel", kernel)

    kh, kw = kernel.shape[2:]
    bs, ci, h, w = radiance.shape
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
        radiance = radiance[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    # -------------------------------------------------------------------------
    # Vectorize the kernel tiles
    kernel = kernel.permute(0, 2, 3, 1)
    kernel = kernel.contiguous().view(bs, 1, kh, kw, ksize*ksize)

    # Split the input buffer in tiles matching the kernels
    tiles = radiance.unfold(2, ksize, 1).unfold(3, ksize, 1)
    tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)

    mask_tiles = mask.unfold(2, ksize, 1).unfold(3, ksize, 1)
    mask_tiles = mask_tiles.contiguous().view(bs, 1, kh, kw, ksize*ksize)
    # -------------------------------------------------------------------------

    # Mask the kernels
    kernel = kernel*mask_tiles

    # Weight local samples and integrate
    weighted_sum = th.sum(kernel*tiles, dim=4)
    kernel_sum = th.sum(kernel, dim=4)

    if kernel_list is not None:
      kernel_list.append(kernel)

    return weighted_sum, kernel_sum

  def forward(self, samples, kernel_list=None):
    radiance = samples["radiance"]
    features = samples["features"]
    gfeatures = samples["global_features"]

    bs, spp, ci, h, w = radiance.shape
    ksize = self.ksize

    sample_count = radiance.new().detach()
    sample_count.resize_(bs, 1, h, w)
    sample_count.zero_()

    s = None
    bs, spp = radiance.shape[:2]
    if self.randomize_spp and self.training:
      s = np.random.randint(1, spp+1)
      radiance = radiance[:, :s, ...]
      features = features[:, :s, ...]
      spp = s

    if gfeatures is not None:
      gfeatures = gfeatures.repeat([1, 1, h, w])

    # Actual RNN evaluation
    self.imdebug.step()
    for step in range(spp):
      f = self.get_features(step, features, gfeatures)
      r = radiance[:, step, ...]

      if step == 0:
        # Zero state
        state = self.main_processor.get_init_state(features[:, 0])

        # All samples
        mask = radiance.new()
        mask.resize_(bs, 1, h, w)
        mask.fill_(1.0)

      sample_count += mask

      # mask features, radiance etc
      f = f*mask
      r = r*mask

      self.imdebug.update("mask_{}".format(step), mask)

      # Advance state (features, mask, state) -> rnn_out, state
      rnn_features, state = self.main_processor(th.cat([f, mask], 1), state)

      # (rnn_out, input, mask) -> kernel, out
      weighted_sum, kernel_sum = self.predict_and_apply_kernels(
          rnn_features, f, r, mask, kernel_list)


      # (rnn_features, mask) -> variance
      log_variance = self.variance_regressor(th.cat([rnn_features, mask], 1))
      log_variance = th.clamp(log_variance, -10, 10)  # prevents diverging loss

      if step == 0:
        # Initialize output buffers
        output = weighted_sum
        sum_weights = kernel_sum
      else:
        # Update unnormalized output, and normalizer
        output = output + weighted_sum
        sum_weights = sum_weights + kernel_sum

      # output, weights, variance -> new sampling mask
      mask = self.get_sampling_mask(sum_weights, output, log_variance)

    # Normalize output
    output = output / (sum_weights + self.eps)

    log_variance = crop_like(log_variance, output)
    sample_count = crop_like(sample_count, output)
    # output = crop_like(output, log_variance)

    return output, log_variance, sum_weights, sample_count

  def get_sampling_mask(self, sum_weights, output, log_variance):
    with th.no_grad():
      # TODO: handle crop?
      # log_variance = crop_like(log_variance, output)
      variance = th.exp(log_variance)
      output = output / (sum_weights + self.eps)
      padding = np.array(variance.shape[-2:]) - np.array(output.shape[-2:])

      variance = crop_like(variance, output)

      pads = []
      for i, p in enumerate(padding):
        pads.append(int(p) // 2)
        pads.append(int(p - p // 2))
      output = F.pad(output, pads, "constant", 0)
      variance = F.pad(variance, pads, "constant", 0)

      eps = 1e-2
      denom = np.square(output).mean(1, keepdim=True) + eps*eps

      # Laplace distribution (variance = 2*b^2)
      estimator = 2*np.square(variance) / denom

      bs, _, h, w = log_variance.shape
      n = h*w
      count = int(self.ratio * n)

      mask = log_variance.new()
      mask.resize_(log_variance.shape)
      mask.zero_()
      for b in range(bs):
        p = estimator[b].view(-1)
        denom = p.sum()
        if denom > 0:
          p = p / denom
        sample_idx = np.random.choice(
            range(n), size=(count, ), replace=False, p = p.cpu().numpy())
        samples  = np.zeros((h*w, ))
        samples[sample_idx] += 1
        mask[b] = th.from_numpy(np.reshape(samples, [1, 1, h, w])).cuda()
      return mask 

  def get_features(self, step, features, gfeatures):
    if gfeatures is not None:
      f = [features[:, step, ...]]
      if gfeatures is not None:
        f.append(gfeatures)
      f = th.cat(f, 1)
    else:
      f = features[:, step, ...]
    return f

class DisneyKPAE(nn.Module):
  def __init__(self, pre, nfeats=27, ksize=21, depth=9, width=100, bayesian=False):
    super(DisneyKPAE, self).__init__()

    self.ksize = ksize

    self.pre = pre

    self.diffuse = ConvChain(
        self.pre.n_px_features, ksize*ksize, depth=depth, 
        width=width, ksize=5,
        activation="relu", weight_norm=False,
        pad=False, output_type="linear")

    self.specular = ConvChain(
        self.pre.n_px_features, ksize*ksize, depth=depth, 
        width=width, ksize=5,
        activation="relu", weight_norm=False,
        pad=False, output_type="linear")

    self.softmax = nn.Softmax2d()


  def forward(self, data, kernel_list=None):
    # softmax-normalized outputs
    k_diffuse = self.softmax(self.diffuse(data["diffuse_in"]))
    k_specular = self.softmax(self.specular(data["specular_in"]))

    if kernel_list is not None:
      kernel_list.append(k_diffuse)

    # kernel reconstruction
    r_diffuse, _ = apply_kernels(k_diffuse, data["diffuse_buffer"], normalization="none")
    r_specular, _ = apply_kernels(k_specular, data["specular_buffer"], normalization="none")

    albedo = crop_like(data["albedo"], r_diffuse)
    final_specular = th.exp(r_specular) - 1
    final_diffuse = albedo * r_diffuse

    final_radiance = final_diffuse + final_specular

    output = {
        "radiance": final_radiance,
        "diffuse": r_diffuse,
        "specular": r_specular,
        }

    return output

class DirectSamples(nn.Module):
  def __init__(self, pre, width=128, bayesian=False, randomize_spp=True):
    super(DirectSamples, self).__init__()

    self.randomize_spp = randomize_spp

    self.pre = pre

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures
    self.embedding = ConvChain(
        nfeats, width, depth=7, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.decoder = ConvChain(
        width, 3, depth=3, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

  def forward(self, samples, kernel_list=None):
    features = samples["features"]
    gfeatures = samples["global_features"]

    bs, spp, _, h, w = features.shape
    gfeatures = gfeatures.repeat([1, 1, h, w])
    s = spp
    if self.randomize_spp and self.training:
      s = np.random.randint(1, spp+1)
      features = features[:, :s, ...]

    output_feats = []
    for sp in range(s):
      input_feat = th.cat([gfeatures, features[:, sp]], 1)
      output_feats.append(self.embedding(input_feat))

    # average features
    aggregated = sum(output_feats) / s

    output = self.decoder(aggregated)

    return {
        "radiance": output,
        }

class DirectPixels(nn.Module):
  def __init__(self, pre, width=128, bayesian=False, randomize_spp=True):
    super(DirectPixels, self).__init__()

    self.randomize_spp = randomize_spp
    if self.randomize_spp:
      raise ValueError("not supported")

    self.pre = pre

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures
    self.embedding = ConvChain(
        nfeats, width, depth=7, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.decoder = ConvChain(
        width, 3, depth=3, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

  def forward(self, samples, kernel_list=None):
    features = samples["features"]
    gfeatures = samples["global_features"]

    bs, _, h, w = features.shape
    gfeatures = gfeatures.repeat([1, 1, h, w])

    input_feat = th.cat([gfeatures, features], 1)
    output_feats = self.embedding(input_feat)

    output = self.decoder(output_feats)

    return {
        "radiance": output,
        }


class Binning(nn.Module):
  def __init__(self, pre, width=128, ksize=5, bayesian=False, randomize_spp=True):
    super(Binning, self).__init__()

    self.randomize_spp = randomize_spp

    self.pre = pre
    self.ksize = ksize
    self.width = width

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures
    self.nclasses = 4

    self.embedding = ConvChain(
        nfeats, width, depth=7, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="leaky_relu")

    self.classifier = ConvChain(
        width, self.nclasses, depth=3, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.recurrent_state = ConvChain(
        2*width, width, depth=2, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=True, output_type="linear")

    self.kernel_predictor = ConvChain(
        width, ksize*ksize*self.nclasses, depth=2, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.sm = nn.Softmax2d()

  def forward(self, samples, kernel_list=None):
    features = samples["features"]
    radiance = samples["radiance"]
    gfeatures = samples["global_features"]

    nclasses = self.nclasses

    bs, spp, _, h, w = features.shape
    gfeatures = gfeatures.repeat([1, 1, h, w])
    s = spp
    if self.randomize_spp and self.training:
      s = np.random.randint(1, spp+1)
      features = features[:, :s, ...]

    for sp in range(s):
      input_feat = th.cat([gfeatures, features[:, sp]], 1)
      embedded = self.embedding(input_feat)
      classes = self.sm(self.classifier(embedded))

      rad = crop_like(radiance[:, sp], classes)

      w_rad = []
      for c in range(nclasses):
        w_rad.append(rad*classes[:, c:c+1])

      if sp == 0:
        state = embedded
        radiance_classes = th.cat(w_rad, 1) / spp
      else:
        state = self.recurrent_state(th.cat([state, embedded], 1))
        radiance_classes += th.cat(w_rad, 1) / spp

    # predict kernels from final state
    k = self.ksize
    kernels = self.kernel_predictor(state)
    for c in range(nclasses):
      r = radiance_classes[:, 3*c:3*(c+1)]
      ker = self.sm(kernels[:, k*k*c:k*k*(c+1)])
      if kernel_list is not None:
        kernel_list.append(ker)
      if c == 0:
        out = apply_kernels(ker, r, normalization="none")[0]
      else:
        out += apply_kernels(ker, r, normalization="none")[0]

    return {
        "radiance": out,
        }

class FullKP(nn.Module):
  def __init__(self, pre, width=128, ksize=5, spp=4, pixel=False, bayesian=False, randomize_spp=True):
    super(FullKP, self).__init__()

    self.pre = pre
    self.ksize = ksize
    self.spp = spp
    self.pixel = pixel

    nfeats = self.pre.nfeatures
    self.nclasses = 4

    if self.pixel:
      n_in = nfeats
      n_out = ksize*ksize
    else:
      n_in = nfeats * spp
      n_out = ksize*ksize*spp

    n_in += self.pre.n_gfeatures

    self.kernel_predictor = ConvChain(
        n_in, n_out, depth=9, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.sm = nn.Softmax2d()

  def forward(self, samples, kernel_list=None):
    features = samples["features"]
    radiance = samples["radiance"]
    gfeatures = samples["global_features"]

    if self.pixel: # pixel input
      features = features.mean(1)
      radiance = radiance.mean(1)
      bs, c, h, w = features.shape
      gfeatures = gfeatures.repeat([1, 1, h, w])
      input_feat = th.cat([gfeatures, features], 1)
    else:
      # Concatenate all the samples
      bs, spp, c, h, w = features.shape
      gfeatures = gfeatures.repeat([1, 1, h, w])
      assert spp == self.spp
      input_feat = th.cat([gfeatures, features.view(bs, spp*c, h, w)], 1)

    kernels = self.sm(self.kernel_predictor(input_feat))

    if self.pixel:
      kernels = kernels
      if kernel_list is not None:
        kernel_list.append(kernels)
      out = apply_kernels(kernels, radiance, normalization="none")[0]
    else:
      k = self.ksize
      for s in range(spp):
        r = radiance[:, s]
        ker = kernels[:, k*k*s:k*k*(s+1)]
        if kernel_list is not None:
          kernel_list.append(ker)
        if s == 0:
          out = apply_kernels(ker, r, normalization="none")[0]
        else:
          out += apply_kernels(ker, r, normalization="none")[0]

    return {
        "radiance": out,
        }

class MaxPooled(nn.Module):
  def __init__(self, pre, width=128, ksize=21, bayesian=False, randomize_spp=True):
    super(MaxPooled, self).__init__()

    self.randomize_spp = randomize_spp

    self.pre = pre
    self.eps = 1e-8

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures
    self.embedding = ConvChain(
        nfeats, width, depth=7, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="relu")

    self.kernel_regressor = ConvChain(
        width+self.pre.nfeatures, ksize*ksize, depth=3, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

  def forward(self, samples, kernel_list=None):
    features = samples["features"]
    radiance = samples["radiance"]
    gfeatures = samples["global_features"]

    bs, spp, _, h, w = features.shape
    gfeatures = gfeatures.repeat([1, 1, h, w])
    s = spp
    if self.randomize_spp and self.training:
      s = np.random.randint(1, spp+1)
      features = features[:, :s, ...]

    output_feats = []
    for sp in range(s):
      input_feat = th.cat([gfeatures, features[:, sp]], 1)
      output_feat = self.embedding(input_feat)

      if sp == 0:
        aggregated = output_feat
      else:
        aggregated = th.max(aggregated, output_feat)

    for sp in range(s):
      f = features[:, sp]
      r = radiance[:, sp]
      input_feat = th.cat([crop_like(f, aggregated), aggregated], 1)
      kernel = self.kernel_regressor(input_feat)

      if sp == 0:
        max_kernel = kernel.view(bs, -1).max(1)[0].view(bs, 1, 1, 1)
      else:
        old_max = max_kernel
        max_kernel = th.max(old_max, kernel.view(bs, -1).max(1)[0].view(bs, 1, 1, 1))

      # well behaved softmax
      kernel = th.exp(kernel - max_kernel)
      weighted_sum, kernel_sum = apply_kernels(kernel, r, normalization="sum")

      if kernel_list is not None:
        kernel_list.append(kernel)

      if sp == 0:
        unnormalized = weighted_sum
        sum_weights = kernel_sum
      else:
        # make sure all the activities are scaled with the same 'max'
        unnormalized = unnormalized*th.exp(old_max - max_kernel) + weighted_sum
        sum_weights = sum_weights*th.exp(old_max - max_kernel) + kernel_sum

    output = unnormalized / (sum_weights + self.eps)

    return {
        "radiance": output,
        }

class PixelSample(nn.Module):
  """2018-05-17: brute force per sample weight"""

  def __init__(self, pre, pair_width=64, width=128, ksize=5, bayesian=False, randomize_spp=True):
    super(PixelSample, self).__init__()

    self.randomize_spp = randomize_spp

    self.pre = pre
    self.ksize = ksize
    self.width = width
    self.pair_width = pair_width

    nfeats = self.pre.nfeatures + self.pre.n_gfeatures
    self.nclasses = 4

    # fov 1 + 2*depth >= ksize
    # depth >= (ksize - 1) // 2

    self.sample_embedding = ConvChain(
        nfeats, width, depth=7, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.pixel_features = ConvChain(
        width, pair_width, depth=3, 
        width=width, ksize=3,
        activation="leaky_relu", weight_norm=False,
        pad=False, output_type="linear")

    self.sample_features = FullyConnected(
        self.pre.nfeatures, pair_width, depth=1, 
        width=pair_width) 

    self.kernel_predictor = FullyConnected(
        pair_width, 1, depth=3, 
        width=pair_width) 

  def forward(self, samples, kernel_list=None):
    features = samples["features"]
    radiance = samples["radiance"]
    gfeatures = samples["global_features"]

    bs, spp, nf, h, w = features.shape
    gfeatures = gfeatures.repeat([1, 1, h, w])

    for sp in range(spp):
      input_feat = th.cat([gfeatures, features[:, sp]], 1)
      embed = self.sample_embedding(input_feat)
      if sp == 0:
        pixel_embed = embed
      else:  # Max-pool over samples
        pixel_embed = th.max(pixel_embed, embed)

    pixel_features = self.pixel_features(pixel_embed)

    ph, pw = pixel_features.shape[2:]
    # Crop kernel and input so their sizes match
    ksize = self.ksize
    needed = ph + ksize - 1
    if needed > h:
      crop = (needed - h) // 2
      if crop > 0:
        pixel_features = pixel_features[:, :, crop:-crop, crop:-crop]
      ph, pw = pixel_features.shape[2:]
    else:
      crop = (h - needed) // 2
      if crop > 0:
        features = features[:, :, :, crop:-crop, crop:-crop]
        radiance = radiance[:, :, :, crop:-crop, crop:-crop]

    # -------------------------------------------------------------------------

    # Split the samples buffer in tiles matching the pixel_features
    tiles = features.unfold(3, ksize, 1).unfold(4, ksize, 1)
    # bs, spp ,nf, ph, pw, ksize_h, ksize_w
    # print(tiles.shape)
    tiles = tiles.permute(0, 3, 4, 1, 5, 6, 2).contiguous()
    # bs, ph, pw, spp, ksize_h, ksize_w, nf
    # print(tiles.shape)
    tiles = tiles.view(bs*ph*pw, spp*ksize*ksize, nf)
    # print(tiles.shape)
    
    tiles_r = radiance.unfold(3, ksize, 1).unfold(4, ksize, 1)
    # bs, spp, 3, ph, pw, ksize_h, ksize_w
    # print(tiles_r.shape)
    tiles_r = tiles_r.permute(0, 2, 3, 4, 1, 5, 6).contiguous()
    # bs, 3, ph, pw, spp, ksize_h, ksize_w
    # print(tiles_r.shape)
    tiles_r = tiles_r.view(bs, 3, ph, pw, spp*ksize*ksize)

    pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous().view(bs*ph*pw, self.pair_width)

    tfeats = []
    tradiance = []
    for s in range(spp):
      for ky in range(ksize):
        for kx in range(ksize):
          s_idx = kx + ksize*(ky + ksize * s)

          # Add kernel-wise xy coordinate of the sample
          tile = tiles[:, s_idx]
          x = tile[:, 0:1] + kx - ksize // 2
          y = tile[:, 1:2] + ky - ksize // 2
          tile = th.cat([x, y, tile[:, 2:]], 1)

          sample_features = self.sample_features(tile)
          # TODO: check whether relu kills weights too much
          weights = self.kernel_predictor(pixel_features + sample_features)

          # if s_idx == 0:
          #   max_kernel = kernel.view(bs, -1).max(1)[0].view(bs, 1, 1, 1)
          # else:
          #   old_max = max_kernel
          #   max_kernel = th.max(old_max, kernel.view(bs, -1).max(1)[0].view(bs, 1, 1, 1))

          weights = weights.view(bs, 1, ph, pw)
          weighted_r = weights*tiles_r[..., s_idx]

          if s_idx == 0:
            unnormalized = weighted_r
            sum_weights = weights
          else:
            unnormalized = unnormalized + weighted_r
            sum_weights = sum_weights + weights

    # tfeats = th.cat(tfeats, 1)
    # tradiance = th.cat(tradiance, -1)
    # tfeats = tfeats.view(bs, ph, pw, spp, 1, ksize, ksize)
    # tradiance = tradiance.view(bs, 3, ph, pw, spp, ksize, ksize)
    # tfeats = tfeats[0, 50, 50, :]
    # tradiance = tradiance[0, :, 50, 50, :].permute(1, 0, 2, 3)
    # print(tradiance.shape)
    # import torchlib.debug as D
    # D.tensor(tfeats)
    # D.tensor(tradiance, "tile_radiance")

    output = unnormalized / (sum_weights + 1e-8)

    # print(unnormalized.mean().item(), sum_weights.mean().item(), output.mean().item())

    return {
        "radiance": output,
        }
