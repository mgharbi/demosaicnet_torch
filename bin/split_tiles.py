import argparse
import os
import skimage.io as skio
import numpy as np

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input")
  parser.add_argument("output")
  parser.add_argument("--tile_size", default=32, type=int)
  args = parser.parse_args()

  root = os.path.dirname(args.input)
  idx = 0
  tiles_per_folder = 10000
  with open(args.input) as fid:
    for l in fid.readlines():
      l = l.strip()

      src = os.path.join(root, l)
      im = skio.imread(src)
      h, w = im.shape[:2]
      tiles = np.split(im, h // args.tile_size, 0)
      for t in tiles:
        tiles2 = np.split(t, w // args.tile_size, 1)
        for t2 in tiles2:
          dir_id = "{:04d}".format(idx // tiles_per_folder)
          dirname = os.path.join(args.output, dir_id)
          os.makedirs(dirname, exist_ok=True)
          filename = os.path.join(dirname, "{:010d}.png".format(idx))
          skio.imsave(filename, t2)
          idx += 1
      print(dirname, idx)


if __name__ == "__main__":
  main()
