#!/usr/bin/env python
""""""

import argparse
import logging
import os
import setproctitle
import time

import cv2
import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import skimage.io as skio

from torchlib.trainer import Trainer

import sklearn.cluster as cluster
import sklearn.manifold as manifold

import demosaic.dataset as dset
import demosaic.modules as modules
import demosaic.losses as losses
import demosaic.callbacks as callbacks
import demosaic.converter as converter
import torchlib.callbacks as default_callbacks
import torchlib.optim as toptim
import torchlib.utils as utils
import torchlib.viz as viz

log = logging.getLogger("demosaick")

def main(args):
  linearize = False
  if args.xtrans:
    period = 6
    data = dset.XtransDataset(args.data_dir, transform=None, augment=False, linearize=linearize)
  else:
    period = 2
    data = dset.BayerDataset(args.data_dir, transform=None, augment=False, linearize=linearize)
  loader = DataLoader(
      data, batch_size=args.batch_size, shuffle=True, num_workers=8)

  mask_viz = viz.BatchVisualizer("mask", env="demosaic_inspect")
  mos_viz = viz.BatchVisualizer("mosaic", env="demosaic_inspect")
  diff_viz = viz.BatchVisualizer("diff", env="demosaic_inspect")
  target_viz = viz.BatchVisualizer("target", env="demosaic_inspect")
  input_hist = viz.HistogramVisualizer("color_hist", env="demosaic_inspect")

  for sample in loader:
    mosaic = sample["mosaic"]
    mask = sample["mask"]

    pad = args.ksize // 2
    dx = (pad - args.offset_x) % period
    dy = (pad - args.offset_y) % period
    print("dx {} dy {}".format(dx, dy))
    mosaic = mosaic[..., dy:, dx:]
    mask = mask[..., dy:, dx:]

    def to_patches(arr):
      patches = arr.unfold(2, args.ksize, period).unfold(3, args.ksize, period)
      bs, c, h, w, _, _ = patches.shape
      patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
      patches = patches.view(bs*h*w, c, args.ksize, args.ksize)
      return patches

    patches = to_patches(mosaic)
    bs = patches.shape[0]
    means = patches.view(bs, -1).mean(-1).view(bs, 1, 1, 1)
    std = patches.view(bs, -1).std(-1).view(bs, 1, 1, 1)
    print(means.min().item(), means.max().item())

    patches -= means
    patches /= std + 1e-8

    new_bs = 1024
    idx = np.random.randint(0, patches.shape[0], (new_bs,))
    patches = patches[idx]

    import torchlib.debug as D
    D.tensor(patches)

    flat = patches.view(new_bs, -1).cpu().numpy()

    nclusts = 16
    clst = cluster.MiniBatchKMeans(n_clusters=nclusts)
    # clst.fit(flat)
    clst_idx = clst.fit_predict(flat)
    colors = np.random.uniform(size=(nclusts, 3))

    manif = manifold.TSNE(n_components=2)
    new_coords = manif.fit_transform(flat)
    color = np.zeros((new_coords.shape[0], 3))
    color = (colors[clst_idx, :]*255).astype(np.uint8)
    print(color.shape)
    D.scatter(th.from_numpy(new_coords[:, 0]), th.from_numpy(new_coords[:, 1]), color=color, key="tsne")

    centers = th.from_numpy(clst.cluster_centers_).view(nclusts, 3, args.ksize, args.ksize)
    D.tensor(centers, "centers")

    for cidx in range(nclusts):
      idx = clst_idx == cidx
      p = th.from_numpy(patches.numpy()[idx])
      D.tensor(p, key="cluster_{:02d}".format(cidx))

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir')

  parser.add_argument('--xtrans', dest="xtrans", action="store_true")
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--offset_x', type=int, default=0)
  parser.add_argument('--offset_y', type=int, default=0)
  parser.add_argument('--ksize', type=int, default=3)
  parser.set_defaults(xtrans=False)

  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)

  main(args)
