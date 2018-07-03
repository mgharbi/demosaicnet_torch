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

log = logging.getLogger("demosaick")

def main(args):
  if args.fix_seed:
    np.random.seed(0)
    th.manual_seed(0)

  meta_params = utils.Checkpointer.get_meta(args.model)

  name = os.path.basename(args.model)

  model = modules.get(meta_params["model"])
  log.info("Model configuration: {}".format(meta_params["model"]))

  checkpointer = utils.Checkpointer(
      args.model, model, None,
      meta_params=meta_params)

  f, _ = checkpointer.load_latest()
  log.info("Loading checkpoint {}".format(f))

  for p in model.parameters():
    p.requires_grad = False

  model.cuda()

  for path in os.listdir(args.data_dir):
    if not os.path.splitext(path)[-1] == ".tif":
      continue
    im = cv2.imread(os.path.join(args.data_dir, path), -1).astype(np.float32)
    im = im[args.offset_y:, args.offset_x:]
    sz = 512
    im = im[args.shift_x:args.shift_x+sz, args.shift_y:args.shift_y+sz]
    im /= 2**16
    im = np.stack([im]*3, 0)
    im = dset.bayer_mosaic(im)
    im = np.expand_dims(im, 0)
    im = th.from_numpy(im).cuda()

    h, w = im.shape[-2:]

    tot_time = 0

    sample = {"mosaic": im}

    th.cuda.synchronize()
    start = time.time()

    kernels = []
    out = model(sample, kernel_list=kernels)

    r_kernels = []
    for k in kernels:
      bs, ksz, _, kh, kw = k.shape
      k = k.permute([0, 3, 1, 4, 2]).contiguous().view(bs, 1, kh*ksz, kw*ksz)
      r_kernels.append(k)

    r_kernels = th.cat(r_kernels, 1).squeeze().permute([1, 2, 0])

    th.cuda.synchronize()
    tot_time += time.time()-start

    oh, ow = out.shape[2:]
    ch = (h-oh) // 2
    cw = (w-ow) // 2
    
    im = im[:, ch:-ch, cw:-cw]

    out = out.squeeze(0)

    tot_time *= 1000
    log.info("Time {:.0f} ms".format(tot_time))

    out = out.permute(1, 2, 0).cpu().numpy()
    out = np.clip(out, 0, 1)

    os.makedirs(args.output, exist_ok=True)
    dst_path = os.path.join(args.output, path.replace(".tif", "_"+name+"_.png"))
    skio.imsave(dst_path, out)

    r_kernels /= r_kernels.abs().max()
    r_kernels *= 0.5
    r_kernels += 0.5
    # r_kernels = th.clip(r_kernels, 0, 1)
    r_kernels = r_kernels.cpu().numpy()

    dst_path = os.path.join(args.output, path.replace(".tif", "_kernels.png"))
    skio.imsave(dst_path, r_kernels)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('model')
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--shift_x', default=0, type=int)
  parser.add_argument('--shift_y', default=0, type=int)
  parser.add_argument('--offset_x', default=0, type=int)
  parser.add_argument('--offset_y', default=0, type=int)
  parser.add_argument('--tile_size', default=2048, type=int)
  parser.add_argument('--tile_step', default=1920, type=int)

  # Monitoring
  parser.add_argument('--debug', dest="debug", action="store_true")

  parser.set_defaults(debug=False, fix_seed=False)

  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  # setproctitle.setproctitle(
  #     'demosaic_{}'.format(os.path.basename(args.output)))

  main(args)
