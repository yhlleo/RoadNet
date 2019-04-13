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

BATCH = 16

class Model(GANModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, None, 4], 'anchor'),
                InputDesc(tf.float32, [None, None, None, 4], 'positive'),
                InputDesc(tf.float32, [None, None, None, 4], 'negative')]

    @auto_reuse_variable_scope
    def generator(self, inputs):
        with argscope(Conv2D, kernel_shape=3, stride=1, nl=SeLU):
            l = Conv2D('conv1_1', inputs, 32)
            l = Conv2D('conv1_2', l, 32)
            l = MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 64)
            l = Conv2D('conv2_2', l, 64)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 128)
            l = Conv2D('conv3_2', l, 128)
            l = Conv2D('conv3_3', l, 128)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 256)
            l = Conv2D('conv4_2', l, 256)
            l = Conv2D('conv4_3', l, 256)
            l = MaxPooling('pool4', l, 2)

            l = Conv2D('conv5_1', l, 256)
            l = Conv2D('conv5_2', l, 256)
            l = Conv2D('conv5_3', l, 256)
            l = MaxPooling('pool5', l, 2) # 8x8
            #l = LayerNorm('lrn', l)

        l = GlobalAvgPooling('gap', l)
        l = Dropout('drop5', l, keep_prob=0.8)
        fc6 = FullyConnected('fc6', l, out_dim=2048, nl=tf.identity)
        fc6 = Dropout('drop6', fc6, keep_prob=0.5)
        fc7 = FullyConnected('fc7', fc6, out_dim=256, nl=tf.identity)
        return fc7#tf.nn.softmax(fc7, name='output')

    @auto_reuse_variable_scope
    def discriminator(self, inputs, outputs):
        """ return a (b, 1) logits"""
        l = tf.concat([inputs, outputs], 1)
        with argscope(FullyConnected, nl=tf.identity), \
                argscope(LeakyReLU, alpha=0.2), \
                argscope(BatchNorm, decay=0.9997, epsilon=1e-3):
            l = FullyConnected('fc1', l, out_dim=2048, nl=LeakyReLU)
            l = Dropout('drop1', l, keep_prob=0.5)
            l = FullyConnected('fc2', l, out_dim=2048, nl=LeakyReLU)
            l = Dropout('drop2', l, keep_prob=0.5)
            l = FullyConnected('fc3', l, out_dim=1, nl=tf.identity)
        return l

    def _build_graph(self, inputs):
        anchor, positive, negative = inputs
        with argscope([Conv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                anchor_output = self.generator(anchor)
                positive_output = self.generator(positive)
                negative_output = self.generator(negative)
            with tf.variable_scope('discrim'):
                real_pred = self.discriminator(anchor_output, positive_output)
                fake_pred = self.discriminator(anchor_output, negative_output)

        if get_current_tower_context().is_training:

            t_loss = triplet_loss(
                anchor_output, 
                positive_output, 
                negative_output,
                0.2)

            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            
            cost = tf.add(t_loss, wd_cost, name='cost')
            add_moving_summary(t_loss, cost)

            self.build_losses(real_pred, fake_pred)
            self.g_loss = tf.add(self.g_loss*0.1, cost, name='total_g_loss')
            add_moving_summary(self.g_loss)

            self.collect_variables()

        viz = tf.concat([anchor_output, positive_output, negative_output], 0, name='viz')
        viz2 = tf.concat([real_pred, fake_pred], 0, name='viz2')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def split_input(inputs):
    '''
    inputs: [image, line]:
      - image: an RGB image of shape (s, 3s, 3)
      - line: a gray scale image of shape (s, 3s)
    '''
    #print inputs
    image, line = inputs
    s = image.shape[0]
    assert image.shape[1] == 3*s
    anchor = np.concatenate((image[:,:s,:], line[:,:s][:,:,np.newaxis]), axis=2)
    #print type(anchor_im[0,0,0]), type(anchor_ln[0,0,0])
    positive = np.concatenate((image[:, s:2*s, :], line[:,s:2*s][:,:,np.newaxis]),axis=2)
    negative = np.concatenate((image[:, 2*s:, :], line[:,2*s:][:,:,np.newaxis]),axis=2)
    return [anchor, positive, negative]


def get_data(name):
    isTrain = name == 'train'
    ds = dataset.LineMatchSet(name, shuffle=True)

    ds = MapData(ds, lambda dp: split_input(dp))
    ds = BatchData(ds, BATCH)

    ds = PrefetchData(ds, 100, 1)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=2),
            ScheduledHyperParamSetter('learning_rate', [(300, 8e-5), (400, 2e-5), (400, 1e-5)]),
            HumanHyperParamSetter('learning_rate')
        ],
        model=Model(),
        steps_per_epoch=dataset_train.size(),
        max_epoch=500,
    )


def sample(model_path, image_path, line_path, output):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['anchor', 'positive', 'negative'],
        output_names=['viz', 'viz2'])
            
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    line = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
    ds = dataset.LineSample(image, line)

    ds = MapData(ds, lambda dp: split_input(dp))
    ds = BatchData(ds, 1)
    pred = SimpleDatasetPredictor(pred, ds)
    fname = os.path.basename(image_path).split('.')[0]
    for p in pred.get_result():
        p = p[0]
        with open(os.path.join(output, fname+'.txt'), 'w') as fp:
            np.savetxt(fp, p, fmt='%.8f')
            fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--image', help='image directory')
    parser.add_argument('--line', help='line directory')
    parser.add_argument('--output', help='output file path')
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.sample:
        sample(args.load, args.image, args.line, args.output)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
