#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

sys.path.insert(0, '../../tensorpack')

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

#TOTAL_BATCH_SIZE = 256
DEPTH = 18

class Model(ModelDesc):
    def __init__(self, n=5, data_format='NCHW'):
        self.n = n
        self.data_format = data_format

    def _get_inputs(self):
        # uint8 instead of float32 is used as input type to reduce copy overhead.
        # It might hurt the performance a liiiitle bit.
        # The pretrained models were trained with float32.
        return [InputDesc(tf.uint8, [None, 128, 256, 4], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        #image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        #image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        #image = (image - image_mean) / image_std
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .BNReLU('bnlast')
                      .GlobalAvgPooling('gap')
                      .FullyConnected('fc4', 1000, nl=tf.identity)
                      .FullyConnected('fc5', 2, nl=tf.identity)())
        if get_current_tower_context().is_training:
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
            cost = tf.reduce_mean(cost, name='cross_entropy_loss')

            wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                              480000, 0.2, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            add_moving_summary(cost, wd_cost)
            self.cost = tf.add_n([cost, wd_cost], name='cost')
        
        logits = tf.nn.softmax(logits, name='output')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(name, fake=False):
    isTrain = name == 'train'
    ds = dataset.LinePairs(
        '/home/swoda/tensorpack_data/lineMatch/oxbuilding3', 
        'train', 
        shuffle=True)
    
    if isTrain:
        ds = PrefetchDataZMQ(ds, 8)#min(20, multiprocessing.cpu_count()))
    ds = BatchData(ds, 32, remainder=not isTrain)
    return ds


def get_config(fake=False, data_format='NCHW'):
    dataset_train = get_data('train', fake=fake)
    #dataset_val = get_data('val', fake=fake)

    return TrainConfig(
        model=Model(data_format=data_format),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-3), (60, 5e-4), (85, 1e-4), (95, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=600,
        max_epoch=110,
    )


def eval_on_ILSVRC12(model_file, data_dir):
    ds = get_data('val')
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_file),
        input_names=['input'],
        output_names=['output']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.eval:
        BATCH_SIZE = 128    # something that can run on one gpu
        eval_on_ILSVRC12(args.load, args.data)
        sys.exit()

    logger.auto_set_dir()
    config = get_config(fake=args.fake, data_format=args.data_format)
    if args.load:
        config.session_init = SaverRestore(args.load)
    #config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
