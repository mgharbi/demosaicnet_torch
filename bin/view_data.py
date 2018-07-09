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
    data = dset.XtransDataset(args.data_dir, transform=None, augment=True, linearize=linearize)
  else:
    data = dset.BayerDataset(args.data_dir, transform=None, augment=True, linearize=linearize)
  loader = DataLoader(
      data, batch_size=args.batch_size, shuffle=True, num_workers=8)

  if args.xtrans:
    period = 6
  else:
    period = 2

  mask_viz = viz.BatchVisualizer("mask", env="demosaic_inspect")
  mos_viz = viz.BatchVisualizer("mosaic", env="demosaic_inspect")
  diff_viz = viz.BatchVisualizer("diff", env="demosaic_inspect")
  target_viz = viz.BatchVisualizer("target", env="demosaic_inspect")
  input_hist = viz.HistogramVisualizer("color_hist", env="demosaic_inspect")

  for sample in loader:
    mosaic = sample["mosaic"]
    mask = sample["mask"]
    target = sample["target"]

    # for c in [0, 2]:
    #   target[:, c] = 0
    #   mosaic[:, c] = 0

    diff = target-mosaic

    mos_mask = th.cat([mosaic, mask], 1)
    mos_mask = mos_mask.unfold(2, 2*period, 2*period).unfold(3, 2*period, 2*period)
    bs, c, h, w, kh, kw = mos_mask.shape
    mos_mask = mos_mask.permute(0, 2, 3, 1, 4, 5).contiguous().view(bs*h*w, c, kh*kw)

    import ipdb; ipdb.set_trace()

    mask_viz.update(mask)
    mos_viz.update(mosaic)
    diff_viz.update(diff)
    target_viz.update(target)
    input_hist.update(target[:, 1].contiguous().view(-1).numpy())

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir')

  parser.add_argument('--xtrans', dest="xtrans", action="store_true")
  parser.add_argument('--batch_size', type=int, default=64)
  parser.set_defaults(xtrans=False)

  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  log.setLevel(logging.INFO)

  main(args)
