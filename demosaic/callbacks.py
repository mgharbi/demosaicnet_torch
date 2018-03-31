import numpy as np
import torch as th

import torchlib.viz as viz
import torchlib.callbacks as callbacks
from torch.utils.data import DataLoader
import torchlib.utils as utils
from torchlib.image import crop_like

class DemosaicVizCallback(callbacks.Callback):
  def __init__(self, data, model, env=None, batch_size=8, 
               shuffle=False, cuda=True):
    super(DemosaicVizCallback, self).__init__()
    self.batch_size = batch_size
    self.model = model
    self.batch_viz = viz.BatchVisualizer("batch", env=env)
    self._cuda = cuda

    self.loader = DataLoader(
        data, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, drop_last=True)

  # def on_epoch_end(self, epoch, logs):
  def on_batch_end(self, batch, num_batches, logs):
    for batch in self.loader:
      # Get a batch
      batch_v = utils.make_variable(batch, cuda=self._cuda)

      # Forward
      output = self.model(batch_v)

      eps = 1e-8
      mosaic = batch_v["mosaic"].data
      target = batch_v["target"].data
      noise_variance = batch_v["noise_variance"].data
      output = output.data
      target = crop_like(target, output)
      mosaic = crop_like(mosaic, output)

      vizdata = th.cat( [mosaic, output, target], 0)
      vizdata = np.clip(vizdata.cpu().numpy(), 0, 1)

      # Display
      self.batch_viz.update(
          vizdata, per_row=self.batch_size, 
          caption="{} | input, ours, reference".format(self.current_epoch))

      return  # process only one batch
