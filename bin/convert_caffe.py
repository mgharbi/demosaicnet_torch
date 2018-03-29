#!/usr/bin/env python
""""""

import argparse
import os
import caffe
import json
import numpy as np

def main():
  nets = ["bayer", "bayer_noise", "xtrans"]
  for name in nets:
    path = os.path.join("pretrained_models", name)
    weights = os.path.join(path, "weights.caffemodel")
    proto = os.path.join(path, "deploy.prototxt")

    net = caffe.Net(proto, caffe.TEST)
    net.copy_from(weights)

    output_path = os.path.join("pretrained_models", "numpy", name)
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    for item in net.params.items():
      layer_name, layer = item
      print('convert layer: ' + layer_name)

      num = 0
      for p in net.params[layer_name]:
        np.save(output_path + '/' + str(layer_name) + '_' + str(num), p.data)
        num += 1

if __name__ == "__main__":
  main()
