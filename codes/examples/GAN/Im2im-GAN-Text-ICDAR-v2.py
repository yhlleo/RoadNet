#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Image2Image.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import numpy as np
import tensorflow as tf
import glob
import pickle
import os
import sys
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.symbolic_functions import *
from GAN import GANTrainer, GANModelDesc

"""
To train Image-to-Image translation model with image pairs:
    ./Image2Image.py --data /path/to/datadir --mode {AtoB,BtoA}
    # datadir should contain jpg images of shpae 2s x s, formed by A and B
    # you can download some data from the original authors:
    # https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

Speed:
    On GTX1080 with BATCH=1, the speed is about 9.3it/s (the original torch version is 9.5it/s)

Training visualization will appear be in tensorboard.
To visualize on test set:
    ./Image2Image.py --sample --data /path/to/test/datadir --mode {AtoB,BtoA} --load model

"""

BATCH = 1
IN_CH = 3
OUT_CH = 1
LAMBDA = 100
BETA = 10
ALPHA = 10
SCALE = 2
NF = 64  # number of filter
SHAPE_H, SHAPE_W = 384, 512

class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SHAPE_H, SHAPE_W, IN_CH), 'input'),
                InputDesc(tf.float32, (None, SHAPE_H // SCALE , SHAPE_W // SCALE, OUT_CH), 'edge'),
                InputDesc(tf.float32, (None, SHAPE_H // SCALE, SHAPE_W // SCALE, OUT_CH), 'region'),
                InputDesc(tf.float32, (None, SHAPE_H, SHAPE_W, OUT_CH), 'output')]

    def generator(self, input):
        # imgs: input: 256x256xch
        # U-Net structure, it's slightly different from the original on the location of relu/lrelu
        with argscope(BatchNorm, use_local_stat=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            with argscope(Conv2D, kernel_shape=3, stride=1,
                          nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
                conv1_1 = Conv2D('conv1_1', input, NF, nl=LeakyReLU)      # 384 x 512 x 64
                conv1_2 = Conv2D('conv1_2', conv1_1, NF)                  # 384 x 512 x 64
                pool1 = MaxPooling('pool1', conv1_2, 2)                   # 192 x 256 x 64
                conv2_1 = Conv2D('conv2_1', pool1,  NF * 2)               # 192 x 256 x 128
                conv2_2 = Conv2D('conv2_2', conv2_1,NF * 2)               # 192 x 256 x 128
                pool2 = MaxPooling('pool2', conv2_2, 2)                   # 96  x 128 x 128
                conv3_1 = Conv2D('conv3_1', pool2,NF * 4)                 # 96  x 128 x 256
                conv3_2 = Conv2D('conv3_2', conv3_1, NF * 4)              # 96  x 128 x 256
                conv3_3 = Conv2D('conv3_3', conv3_2, NF * 4)              # 96  x 128 x 256
                pool3 = MaxPooling('pool3', conv3_3, 2)                   # 48  x 64  x 256
                conv4_1 = Conv2D('conv4_1', pool3, NF * 8)                # 48  x 64  x 512
                conv4_2 = Conv2D('conv4_2', conv4_1, NF * 8)              # 48  x 64  x 512
                conv4_3 = Conv2D('conv4_3', conv4_2, NF * 8)              # 48  x 64  x 512
                pool4 = MaxPooling('pool4', conv4_3, 2)                   # 24  x 32  x 512
                conv5_1 = Conv2D('conv5_1', pool4, NF * 8)                # 24  x 32  x 512
                conv5_2 = Conv2D('conv5_2', conv5_1, NF * 8)              # 24  x 32  x 512
                conv5_3 = Conv2D('conv5_3', conv5_2, NF * 8)              # 24  x 32  x 512
                pool5 = MaxPooling('pool5', conv5_3, 2)                   # 12  x 16  x 512
                conv6 = Conv2D('conv6', pool5, NF * 8, nl=BNReLU)         # 12  x 16  x 512
            # edge prediction
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                edge = (LinearWrap(conv6)                    # 12  x 16
                      .Conv2D('edge/conv1', NF*8, kernel_shape=3, stride=1)
                      .Deconv2D('edge/deconv1', NF * 8)      # 24  x 32
                      .Dropout()
                      .ConcatWith(conv5_3, 3)                  # 24  x 32
                      .Conv2D('edge/conv2', NF*8, kernel_shape=3, stride=1)
                      .Deconv2D('edge/deconv2', NF * 8)      # 48  x 64
                      .Dropout()
                      .ConcatWith(conv4_3, 3)                  # 48  x 64
                      .Conv2D('edge/conv3', NF*4, kernel_shape=3, stride=1)
                      .Deconv2D('edge/deconv3', NF * 4)      # 96  x 128
                      .Dropout()
                      .ConcatWith(conv3_3, 3)                  # 96  x 128
                      .Conv2D('edge/conv4', NF*2, kernel_shape=3, stride=1)
                      .Deconv2D('edge/deconv4', NF * 2)      # 192 x 256
                      .Dropout()
                      .ConcatWith(conv2_2, 3) 
                      .Conv2D('edge/conv5', NF, kernel_shape=3, stride=1) # 192 x 256
                      .Conv2D('edge/conv6', OUT_CH, kernel_shape=3, stride=1, nl=tf.sigmoid)()) # 192 x 256
            # region prediction
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                region = (LinearWrap(conv6)                  # 12  x 16
                      .Conv2D('region/conv1', NF*8, kernel_shape=3, stride=1)
                      .Deconv2D('region/deconv1', NF * 8)    # 24  x 32
                      .Dropout()
                      .ConcatWith(conv5_3, 3)                  # 24  x 32
                      .Conv2D('region/conv2', NF*8, kernel_shape=3, stride=1)
                      .Deconv2D('region/deconv2', NF * 8)    # 48  x 64
                      .Dropout()
                      .ConcatWith(conv4_3, 3)                  # 48  x 64
                      .Conv2D('region/conv3', NF*4, kernel_shape=3, stride=1)
                      .Deconv2D('region/deconv3', NF * 4)    # 96  x 128
                      .Dropout()
                      .ConcatWith(conv3_3, 3)                  # 96  x 128
                      .Conv2D('region/conv4', NF*2, kernel_shape=3, stride=1)
                      .Deconv2D('region/deconv4', NF * 2)    # 192 x 256
                      .Dropout()
                      .ConcatWith([conv2_2, edge], 3) 
                      .Conv2D('region/conv5', NF, kernel_shape=3, stride=1) # 192 x 256
                      .Conv2D('region/conv6', OUT_CH, kernel_shape=3, stride=1, nl=tf.sigmoid)()) # 192 x 256
            # char segmentation
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                output = (LinearWrap(conv6)              # 12  x 16
                      .Conv2D('char/conv1_1', NF*8, kernel_shape=3, stride=1)
                      .Conv2D('char/conv1_2', NF*8, kernel_shape=3, stride=1)
                      .Deconv2D('char/deconv1', NF * 8)  # 24  x 32
                      .Dropout()
                      .ConcatWith(conv5_3, 3)              # 24  x 32
                      .Conv2D('char/conv2_1', NF*8, kernel_shape=3, stride=1)
                      .Conv2D('char/conv2_2', NF*8, kernel_shape=3, stride=1)
                      .Deconv2D('char/deconv2', NF * 8)  # 48  x 64
                      .Dropout()
                      .ConcatWith(conv4_3, 3)              # 48  x 64
                      .Conv2D('char/conv3_1', NF*4, kernel_shape=3, stride=1)
                      .Conv2D('char/conv3_2', NF*4, kernel_shape=3, stride=1)
                      .Deconv2D('char/deconv3', NF * 4)  # 96  x 128
                      .Dropout()
                      .ConcatWith(conv3_3, 3)  # 96  x 128
                      .Conv2D('char/conv4_1', NF*2, kernel_shape=3, stride=1)
                      .Conv2D('char/conv4_2', NF*2, kernel_shape=3, stride=1)
                      .Deconv2D('char/deconv4', NF * 2)      # 192 x 256
                      .Dropout()
                      .ConcatWith([conv2_2, edge, region], 3)
                      .Conv2D('char/conv5_1', NF, kernel_shape=3, stride=1)
                      .Conv2D('char/conv5_2', NF, kernel_shape=3, stride=1)
                      .Deconv2D('char/deconv5', NF)
                      .Dropout()
                      .ConcatWith(conv1_2, 3)  # 96  x 128
                      .Conv2D('char/conv6_1', NF, kernel_shape=3, stride=1)
                      .Conv2D('char/conv6_2', OUT_CH, kernel_shape=3, stride=1, nl=tf.sigmoid)())  # 384 x 512
            return edge, region, output

    @auto_reuse_variable_scope
    def discriminator(self, input, output):
        """ return a (b, 1) logits"""
        l = tf.concat([input, output], 3)
        with argscope(Conv2D, nl=tf.identity, kernel_shape=3, stride=2):
            l = (LinearWrap(l)                                  # 512 x 384
                 .Conv2D('conv0_1', NF, stride=1, nl=LeakyReLU)
                 .Conv2D('conv0_2', NF)                         # 256 x 192
                 .BatchNorm('bn0').LeakyReLU()
                 .Conv2D('conv1_1', NF * 2, stride=1, nl=LeakyReLU)
                 .Conv2D('conv1_2', NF * 2)                     # 128 x 96
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2_1', NF * 4, stride=1, nl=LeakyReLU)
                 .Conv2D('conv2_2', NF * 4)                     # 64  x 48
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3_1', NF * 8, stride=1, nl=LeakyReLU)
                 .Conv2D('conv3_2', NF * 8)                     # 32  x 24
                 .BatchNorm('bn3').LeakyReLU()
                 .Conv2D('convlast', 1, stride=1, padding='VALID')()) # 32 x 24
        return l

    def _build_graph(self, inputs):
        input, edge, region, output = inputs
        edge, region, output = edge / 255., region / 255., output / 255.

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)), \
                argscope(LeakyReLU, alpha=0.2):
            with tf.variable_scope('gen'):
                fake_edge, fake_region, fake_output = self.generator(input)
                edge_loss = class_balanced_sigmoid_cross_entropy(edge, fake_edge, name="edge_loss")
                region_loss = class_balanced_sigmoid_cross_entropy(region, fake_region, name="region_loss")
                aux_loss = tf.add(edge_loss, region_loss, name='aux_loss')
                add_moving_summary(edge_loss, region_loss, aux_loss)
                
                errL1 = tf.reduce_mean(tf.abs(fake_output - output), name='L1_loss')
                cb_loss = class_balanced_sigmoid_cross_entropy(output, fake_output, name="cb_loss")
                seg_loss = tf.add(errL1, cb_loss*BETA, name='seg_loss')
                add_moving_summary(errL1, cb_loss, seg_loss)
                gen_loss = tf.add(aux_loss*BETA, seg_loss*LAMBDA, name='gen_loss')
                add_moving_summary(gen_loss)

            with tf.variable_scope('discrim'):
                real_pred = self.discriminator(input, output)
                fake_pred = self.discriminator(input, fake_output)

        self.build_losses(real_pred, fake_pred)
        self.g_loss = tf.add(self.g_loss, gen_loss, name='total_g_loss')
        add_moving_summary(self.g_loss)

        # tensorboard visualization
        if IN_CH == 1:
            input = tf.image.grayscale_to_rgb(input)
        if OUT_CH == 1:
            fake_output = fake_output * 255.
            input = tf.image.rgb_to_grayscale(input)
        viz = tf.concat([input, fake_output], 2) #* 255.0 #+ 1.0) * 128.0
        viz = tf.cast(viz, tf.uint8, name='viz')
        tf.summary.image('input, fake', viz, max_outputs=max(30, BATCH))

        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-5, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

def bin2edge(im):
  s = im.shape
  edge = np.zeros([s[0], s[1]], dtype=np.uint8)
  for i in xrange(1,s[0]-1):
    for j in xrange(1, s[1]-1):
      if im.item(i,j) != im.item(i-1,j) or \
         im.item(i,j) != im.item(i,j-1) or \
         im.item(i,j) != im.item(i-1,j-1) or \
         im.item(i,j) != im.item(i+1,j) or \
         im.item(i,j) != im.item(i,j+1) or \
         im.item(i,j) != im.item(i-1,j+1) or \
         im.item(i,j) != im.item(i+1,j-1) or \
         im.item(i,j) != im.item(i+1,j+1):
        edge.itemset(i, j, 255)
  return edge

def split_input(img):
    """
    img: an RGB image of shape (h, 3w, 3).
    :return: [input, region, output]
    """
    # split the image into left + right pairs
    w = SHAPE_W #img.shape[0]
    assert img.shape[1] == 3 * w
    input, region, output = img[:, :w, :], img[:, w:2*w, :], img[:,2*w:, :]
    if args.mode == 'BtoA':
        input, output = output, input
    if IN_CH == 1:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    if OUT_CH == 1:
        region = cv2.resize(region[:,:,0], (output.shape[1] // SCALE, output.shape[0] // SCALE))[:, :, np.newaxis]
        edge = cv2.resize(bin2edge(output[:,:,0]), (output.shape[1] // SCALE, output.shape[0] // SCALE))[:, :, np.newaxis]
        output = output[:,:,0][:, :, np.newaxis]
    return [input, edge, region, output]

def sample_split_input(img):
    """
    img: an RGB image of shape (h, w, 3).
    :return: [input, region, output]
    """
    # split the image into left + right pairs
    s = img.shape
    edge = np.zeros([s[0], s[1], 3], dtype=np.uint8)
    region = np.zeros([s[0], s[1], 3], dtype=np.uint8)
    output = np.zeros([s[0], s[1], 3], dtype=np.uint8)
    input = img
    if args.mode == 'BtoA':
        input, output = output, input
    if IN_CH == 1:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    if OUT_CH == 1:
        output = output[:,:,0][:, :, np.newaxis]
        region = cv2.resize(region[:,:,0], (output.shape[1] // SCALE, output.shape[0] // SCALE))[:, :, np.newaxis]
        edge = cv2.resize(edge[:,:,0], (output.shape[1] // SCALE, output.shape[0] // SCALE))[:, :, np.newaxis]
    return [input, edge, region, output]


def get_data():
    datadir = args.data
    imgs = glob.glob(os.path.join(datadir, '*.png'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)

    ds = MapData(ds, lambda dp: split_input(dp[0]))
    #augs = [imgaug.Resize(80), imgaug.RandomCrop(64)]
    #ds = AugmentImageComponents(ds, augs, (0, 1))
    ds = BatchData(ds, BATCH)
    ds = PrefetchData(ds, 100, 1)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset = get_data()
    return TrainConfig(
        dataflow=dataset,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=2),
            ScheduledHyperParamSetter('learning_rate', [(100, 1e-5), (200, 5e-6), (300, 1e-6), (400, 8e-7)])
        ],
        model=Model(),
        steps_per_epoch=dataset.size(),
        max_epoch=600,
    )


def sample(datadir, model_path):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input', 'edge', 'region', 'output'],
        output_names=['viz'])

    imgs = glob.glob(os.path.join(datadir, '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = MapData(ds, lambda dp: sample_split_input(dp[0]))
    #ds = AugmentImageComponents(ds, [imgaug.Resize(64)], (0, 1))
    ds = BatchData(ds, 2)

    pred = SimpleDatasetPredictor(pred, ds)
    for o in pred.get_result():
        o = o[0][:, :, :, ::-1]
        stack_patches(o, nr_row=2, nr_col=1, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory', required=True)
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    parser.add_argument('-b', '--batch', type=int, default=1)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch

    if args.sample:
        sample(args.data, args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
