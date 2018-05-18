#!/usr/bin/env python
""""""

import argparse
import logging
import os
import setproctitle
import time

import numpy as np
import torch as th
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from torchlib.trainer import Trainer

import demosaic.dataset as dset
import demosaic.modules as modules
import demosaic.losses as losses
import demosaic.callbacks as callbacks
import demosaic.converter as converter
import torchlib.callbacks as default_callbacks

log = logging.getLogger("demosaick")

def main(args, model_params):
  if args.fix_seed:
    np.random.seed(0)
    th.manual_seed(0)

  # ------------ Set up datasets ----------------------------------------------
  xforms = dset.ToTensor()
  data = dset.BayerDataset(args.data_dir, transform=xforms, augment=True)
  data[0]

  if args.val_data is not None:
    val_data = dset.BayerDataset(args.val_data, transform=xforms, augment=False)
  else:
    val_data = None
  # ---------------------------------------------------------------------------

  model = modules.get(model_params)
  log.info("Model configuration: {}".format(model_params))

  if args.pretrained:
    log.info("Loading Caffe weights")
    cvt = converter.Converter(args.pretrained, model_params["model"])
    cvt.convert(model)

  name = os.path.basename(args.output)
  cbacks = [
      default_callbacks.LossCallback(env=name),
      callbacks.DemosaicVizCallback(val_data, model, cuda=True, 
                                    shuffle=True, env=name),
      ]

  metrics = {
      "psnr": losses.PSNR()
      }

  log.info("Using {} loss".format(args.loss))
  if args.loss == "l2":
    criteria = { "l2": losses.L2Loss(), }
  elif args.loss == "vgg":
    criteria = { "vgg": losses.VGGLoss(), }
  else:
    raise ValueError("not implemented")

  train_params = Trainer.Parameters(
      viz_step=100, lr=args.lr, batch_size=args.batch_size)

  trainer = Trainer(
      data, model, criteria, output=args.output, 
      params = train_params,
      model_params=model_params, verbose=args.debug, 
      callbacks=cbacks,
      metrics=metrics,
      valset=val_data, cuda=True)

  trainer.train()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # I/O params
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--val_data')
  parser.add_argument('--checkpoint')
  parser.add_argument('--pretrained')

  # Training
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--fix_seed', dest="fix_seed", action="store_true")
  parser.add_argument('--loss', default="l2", choices=["l1", "vgg", "l2"])

  # Monitoring
  parser.add_argument('--debug', dest="debug", action="store_true")

  # Model
  parser.add_argument(
      '--params', nargs="*", default=["model=BayerNetwork"])

  parser.set_defaults(debug=False, fix_seed=False)

  args = parser.parse_args()

  params = {}
  if args.params is not None:
    for p in args.params:
      k, v = p.split("=")
      if v.isdigit():
        v = int(v)
      elif v == "False":
        v = False
      elif v == "True":
        v = True

      params[k] = v

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  setproctitle.setproctitle(
      'demosaic_{}'.format(os.path.basename(args.output)))

  main(args, params)
