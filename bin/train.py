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
import demosaic.callbacks as callbacks
import torchlib.callbacks as default_callbacks

log = logging.getLogger("demosaick")

def main(args, model_params):
  if args.fix_seed:
    np.random.seed(0)
    th.manual_seed(0)

  # ------------ Set up datasets ----------------------------------------------
  xforms = dset.ToTensor()
  data = dset.BayerDataset(args.data_dir, transform=xforms)
  data[0]

  if args.val_data is not None:
    val_data = dset.BayerDataset(args.val_data, transform=xforms)
  else:
    val_data = None
  # ---------------------------------------------------------------------------

  model = modules.get(model_params)
  log.info("Model configuration: {}".format(model_params))

  cbacks = [
      default_callbacks.LossCallback(),
      callbacks.DemosaicVizCallback(val_data, model, cuda=True, shuffle=True),
      ]

  metrics = {
      "psnr": modules.PSNR()
      }

  criteria = {
      "l2": modules.L2Loss(),
      }

  train_params = Trainer.Parameters(
      viz_step=10, lr=args.lr, batch_size=args.batch_size)

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

  # Training
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--fix_seed', dest="fix_seed", action="store_true")
  parser.add_argument('--loss', default="l1", choices=["l1"])

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
