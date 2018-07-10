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

  if args.pretrained is not None:
    if args.xtrans:
      model_ref = modules.get({"model": "XtransNetwork"})
      cvt = converter.Converter(args.pretrained, "XtransNetwork")
    else:
      model_ref = modules.get({"model": "BayerNetwork"})
      log.info("Loading Caffe weights")
      cvt = converter.Converter(args.pretrained, "BayerNetwork")
    cvt.convert(model_ref)
    for p in model_ref.parameters():
      p.requires_grad = False
    model_ref.cuda()

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
    # im = im[:3000, :3000]
    im /= 2**16
    im = np.stack([im]*3, 0)
    if args.xtrans:
      im = dset.bayer_mosaic(im)
    else:
      im = dset.xtrans_mosaic(im)
    im = np.expand_dims(im, 0)
    im = th.from_numpy(im).cuda()

    import ipdb; ipdb.set_trace()
    _, _, h, w = im.shape

    out_im = th.zeros(3, h, w).cuda()
    out_ref = th.zeros(3, h, w).cuda()

    tile_size = min(min(args.tile_size, h), w)

    tot_time = 0
    tot_time_ref = 0

    if args.xtrans:
      mod = 6
    else:
      mod = 2

    tile_step = args.tile_step - (args.tile_step % mod)
    tile_step = args.tile_step - (args.tile_step % mod)

    for start_x in range(0, w, tile_step):
      end_x = start_x + tile_size
      if end_x > w:
        # keep mosaic period
        end_x = w
        start_x = end_x - tile_size
        start_x = start_x - (start_x % mod)
        end_x = start_x + tile_size
      for start_y in range(0, h, tile_step):
        end_y = start_y + tile_size
        if end_y > h:
          end_y = h
          start_y = end_y - tile_size
          start_y = start_y - (start_y % mod)
          end_y = start_y + tile_size

        print(start_x, start_y)
        sample = {"mosaic": im[:, :, start_y:end_y, start_x:end_x]}

        th.cuda.synchronize()
        start = time.time()
        out = model(sample)
        th.cuda.synchronize()
        tot_time += time.time()-start

        th.cuda.synchronize()
        start = time.time()
        outr = model_ref(sample)
        th.cuda.synchronize()
        tot_time_ref += time.time()-start

        oh, ow = out.shape[2:]
        ch = (tile_size-oh) // 2
        cw = (tile_size-ow) // 2
        out_im[:, start_y + ch: start_y + ch + oh, start_x + cw: start_x + cw + ow] = out[0]

        oh, ow = outr.shape[2:]
        ch = (tile_size-oh) // 2
        cw = (tile_size-ow) // 2
        out_ref[:, start_y + ch: start_y + ch + oh, start_x + cw: start_x + cw + ow] = outr[0]

    tot_time *= 1000
    tot_time_ref *= 1000
    log.info("Time {:.0f} ms (ref {:.0f} ms)".format(tot_time, tot_time_ref))

    out_im = out_im.permute(1, 2, 0).cpu().numpy()
    out_im = np.clip(out_im, 0, 1)

    os.makedirs(args.output, exist_ok=True)
    dst_path = os.path.join(args.output, path.replace(".tif", "_"+name+"_.png"))
    skio.imsave(dst_path, out_im)

    out_ref = out_ref.permute(1, 2, 0).cpu().numpy()
    out_ref = np.clip(out_ref, 0, 1)

    dst_path = os.path.join(args.output, path.replace(".tif", "_ref.png"))
    skio.imsave(dst_path, out_ref)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('model')
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--offset_x', default=0, type=int)
  parser.add_argument('--offset_y', default=0, type=int)
  parser.add_argument('--tile_size', default=2048, type=int)
  parser.add_argument('--tile_step', default=1920, type=int)
  parser.add_argument('--pretrained')

  # Monitoring
  parser.add_argument('--debug', dest="debug", action="store_true")

  parser.add_argument('--xtrans', dest="xtrans", action="store_true")

  parser.set_defaults(debug=False, fix_seed=False, xtrans=False)

  args = parser.parse_args()

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  setproctitle.setproctitle(
      'demosaic_{}'.format(os.path.basename(args.output)))

  main(args)
