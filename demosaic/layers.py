import numpy as np
import tensorflow as tf

def make_mosaic(im, name='make_mosaic'):
  with tf.name_scope(name):
    mask = np.zeros((2,2,3), dtype=np.float32)
    mask[::2, ::2, 1] = 1
    mask[::2, 1::2, 0] = 1
    mask[1::2, ::2, 2] = 1
    mask[1::2, 1::2, 1] = 1
    sz = tf.shape(im)
  
    if len(im.get_shape().as_list()) == 4:
      mask = mask[np.newaxis, :, :, :]
      mask = tf.tile(mask, [1, sz[1]/2, sz[2]/2, 1])
    elif len(im.get_shape().as_list()) == 3:
      mask = tf.tile(mask, [sz[0]/2, sz[1]/2, 1])
    else:
      raise NotImplemented
    mask.set_shape(im.get_shape())
    ret = im*mask
  return ret

def pack_mosaic(im, name='pack_mosaic'):
  with tf.name_scope(name):
    if len(im.get_shape().as_list()) == 4:
      g0 = tf.expand_dims(im[:, ::2, ::2, 1], 3)
      r  = tf.expand_dims(im[:, ::2, 1::2, 0], 3)
      b  = tf.expand_dims(im[:, 1::2, ::2, 2], 3)
      g1 = tf.expand_dims(im[:, 1::2, 1::2, 1], 3)

      ret = tf.concat([g0, r, b, g1], 3)
    else:
      raise NotImplemented
  return ret

def unpack_mosaic(im, name='unpack_mosaic'):
  with tf.variable_scope(name):
    bs_h_w = tf.shape(im)[:3]
    bs = bs_h_w[0]
    h = bs_h_w[1]
    w = bs_h_w[2]
    nfilters = im.get_shape().as_list()[-1]
    weights = np.zeros([2, 2, 3, nfilters], dtype=np.float32)
    for ochan in range(3):
      weights[0, 0, ochan, 4*ochan+0] = 1.0
      weights[0, 1, ochan, 4*ochan+1] = 1.0
      weights[1, 0, ochan, 4*ochan+2] = 1.0
      weights[1, 1, ochan, 4*ochan+3] = 1.0
    ret = tf.nn.conv2d_transpose(im, weights, [bs, 2*h, 2*w, 3], strides=[1, 2, 2, 1])
    return ret

def upsample(im, factor, name='upsample'):
  with tf.variable_scope(name):
    bs, h, w, nfilters = im.get_shape().as_list()
    weights = np.zeros([1, 1, nfilters, nfilters], dtype=np.float32)
    for i in range(nfilters):
      weights[0, 0, i, i] = 1.0
    ret = tf.nn.conv2d_transpose(im, weights, [bs, factor*h, factor*w, nfilters], strides=[1, factor, factor, 1])
    return ret

