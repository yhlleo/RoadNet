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

sys.path.insert(0, '../../tensorpack')

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
BETA = 20
NF = 32  # number of filter


class Model(GANModelDesc):
    def _get_inputs(self):
        SHAPE = 512
        return [InputDesc(tf.float32, (None, SHAPE, SHAPE, IN_CH), 'input'),
                InputDesc(tf.float32, (None, SHAPE, SHAPE, OUT_CH), 'output')]

    def generator(self, imgs):
        # imgs: input: 256x256xch
        # U-Net structure, it's slightly different from the original on the location of relu/lrelu
        with argscope(BatchNorm, use_local_stat=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            with argscope(Conv2D, kernel_shape=3, stride=1,
                          nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
                conv1_1 = Conv2D('conv1_1', imgs, NF, nl=LeakyReLU)
                conv1_2 = Conv2D('conv1_2', conv1_1, NF) 
                pool1 = MaxPooling('pool1', conv1_2, 2)
                conv2_1 = Conv2D('conv2_1', pool1,  NF * 2)
                conv2_2 = Conv2D('conv2_2', conv2_1,NF * 2)
                pool2 = MaxPooling('pool2', conv2_2, 2)
                conv3_1 = Conv2D('conv3_1', pool2,NF * 4)
                conv3_2 = Conv2D('conv3_2', conv3_1, NF * 4)
                #conv3_3 = Conv2D('conv3_3', conv3_2, NF * 4)
                pool3 = MaxPooling('pool3', conv3_2, 2)
                conv4_1 = Conv2D('conv4_1', pool3, NF * 8)
                conv4_2 = Conv2D('conv4_2', conv4_1, NF * 8)
                #conv4_3 = Conv2D('conv4_3', conv4_2, NF * 8)  
                pool4 = MaxPooling('pool4', conv4_2, 2)
                conv5_1 = Conv2D('conv5_1', pool4, NF * 8)
                conv5_2 = Conv2D('conv5_2', conv5_1, NF * 8)
                #conv5_3 = Conv2D('conv5_3', conv5_2, NF * 8)
                pool5 = MaxPooling('pool5', conv5_2, 2)
                #conv6 = Conv2D('conv6', pool5, NF * 8, kernel_shape=2, nl=BNReLU)
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                return (LinearWrap(pool5) # 1 x 1
                        .Deconv2D('deconv1', NF * 8)
                        .Dropout()
                        .ConcatWith(conv5_2, 3)
                        .Deconv2D('deconv2', NF * 8)
                        .Dropout()
                        .ConcatWith(conv4_2, 3)
                        .Deconv2D('deconv3', NF * 4)
                        .Dropout()
                        .ConcatWith(conv3_2, 3)
                        .Deconv2D('deconv4', NF * 2)
                        .ConcatWith(conv2_2, 3)
                        .Deconv2D('deconv5', NF)
                        .ConcatWith(conv1_2, 3)
                        .Conv2D('deconv6', OUT_CH, nl=tf.sigmoid)())

    @auto_reuse_variable_scope
    def discriminator(self, inputs, outputs):
        """ return a (b, 1) logits"""
        l = tf.concat([inputs, outputs], 3)
        with argscope(Conv2D, nl=tf.identity, kernel_shape=3, stride=2):
            l = (LinearWrap(l)
                 .Conv2D('conv0', NF, nl=LeakyReLU)
                 .Conv2D('conv1', NF * 2)
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', NF * 4)
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', NF * 4)
                 .BatchNorm('bn3').LeakyReLU()
                 .Conv2D('conv4', NF * 8)
                 .BatchNorm('bn4').LeakyReLU()
                 .Conv2D('convlast', 1, stride=1, padding='VALID')())
        return l

    def _build_graph(self, inputs):
        input, output = inputs
        #input, output = input / 128.0 - 1, output / 128.0 - 1
        input, output = input / 255.0, output / 255.0

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)), \
                argscope(LeakyReLU, alpha=0.2):
            with tf.variable_scope('gen'):
                fake_output = self.generator(input)
            with tf.variable_scope('discrim'):
                real_pred = self.discriminator(input, output)
                fake_pred = self.discriminator(input, fake_output)

        self.build_losses(real_pred, fake_pred)
        errL1 = tf.reduce_mean(tf.abs(fake_output - output), name='L1_loss')
        self.g_loss = tf.add(self.g_loss, LAMBDA * errL1, name='total_g_loss')
        add_moving_summary(errL1, self.g_loss)

        # tensorboard visualization
        if IN_CH == 1:
            input = tf.image.grayscale_to_rgb(input)
        if OUT_CH == 1:
            output = tf.image.grayscale_to_rgb(output)
            fake_output = tf.image.grayscale_to_rgb(fake_output)
        viz = (tf.concat([input, output, fake_output], 2)) * 255.0 # + 1.0) * 128.0
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('input,output,fake', viz, max_outputs=max(30, BATCH))

        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 1e-5, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def split_input(img):
    """
    img: an RGB image of shape (s, 2s, 3).
    :return: [input, output]
    """
    # split the image into left + right pairs
    s = img.shape[0]
    assert img.shape[1] == 2 * s
    input, output = img[:, :s, :], img[:, s:, :]
    if args.mode == 'BtoA':
        input, output = output, input
    if IN_CH == 1:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    if OUT_CH == 1:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    return [input, output]


def get_data():
    datadir = args.data
    imgs = glob.glob(os.path.join(datadir, '*.png'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)

    ds = MapData(ds, lambda dp: split_input(dp[0]))
    #augs = [imgaug.Resize(532), imgaug.RandomCrop(512)]
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
        steps_per_epoch=500,#dataset.size(),
        max_epoch=600,
    )


def sample(datadir, model_path, output):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input', 'output'],
        output_names=['viz'])

    imgs = glob.glob(os.path.join(datadir, '*.png'))
    ds = ImageFromFile(imgs, channel=3, shuffle=False)
    ds = MapData(ds, lambda dp: split_input(dp[0]))
    #ds = AugmentImageComponents(ds, [imgaug.Resize(512)], (0, 1))
    ds = BatchData(ds, 1)

    pred = SimpleDatasetPredictor(pred, ds)
    idx = 0
    print pred.get_result()
    for o in pred.get_result():
        o = o[0][0][:,1024:,:]
        fname = imgs[idx].split('/')[-1]
        cv2.imwrite(os.path.join(output, fname), o, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        idx += 1
        #stack_patches(o, nr_row=1, nr_col=1, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory', required=True)
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    parser.add_argument('--output', help='output file path')
    parser.add_argument('-b', '--batch', type=int, default=1)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch

    if args.sample:
        sample(args.data, args.load, args.output)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
