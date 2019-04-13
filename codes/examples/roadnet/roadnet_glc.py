#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: hed.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import tensorflow as tf
import argparse
import numpy as np
from six.moves import zip
import os, glob
import sys
import random

sys.path.insert(0, '../../tensorpack')

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

SHAPE=512

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, SHAPE, SHAPE, 3], 'image'),
                InputDesc(tf.int32, [None, SHAPE, SHAPE], 'segment'),
                InputDesc(tf.int32, [None, SHAPE, SHAPE], 'skeleton')]

    def _build_graph(self, inputs):
        image, segment, skeleton = inputs

        #image = image - tf.constant([104, 116, 122], dtype='float32')
        segment = tf.expand_dims(segment, 3, name='segment4d')
        #boundary = tf.expand_dims(boundary, 3, name='boundary4d')
        skeleton = tf.expand_dims(skeleton, 3, name='skeleton4d')

        def branch(name, l, num, up):
            with tf.variable_scope(name) as scope:
                l = Conv2D('convfc', l, num, kernel_shape=1, nl=tf.identity,
                           use_bias=True,
                           W_init=tf.constant_initializer(),
                           b_init=tf.constant_initializer())
                while up != 1:
                    l = BilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up / 2
                return l

        def network1(name, l, extra=True):
            with argscope(Conv2D, kernel_shape=3, nl=BNSeLU):
                with tf.variable_scope(name):
                    l = Conv2D('conv1_1', l, 64)
                    l = Conv2D('conv1_2', l, 64)
                    b1 = branch('branch1', l, 1, 1)
                    l = MaxPooling('pool1', l, 2)

                    l = Conv2D('conv2_1', l, 128)
                    l = Conv2D('conv2_2', l, 128)
                    b2 = branch('branch2', l, 1, 2)
                    l = MaxPooling('pool2', l, 2)

                    l = Conv2D('conv3_1', l, 256)
                    l = Conv2D('conv3_2', l, 256)
                    if extra:
                        l = Conv2D('conv3_3', l, 256)
                    b3 = branch('branch3', l, 1, 4)
                    l = MaxPooling('pool3', l, 2)

                    l = Conv2D('conv4_1', l, 512)
                    l = Conv2D('conv4_2', l, 512)
                    if extra:
                        l = Conv2D('conv4_3', l, 512)
                    b4 = branch('branch4', l, 1, 8)
                    l = MaxPooling('pool4', l, 2)

                return [b1, b2, b3, b4]

        def network2(name, l):
            with argscope(Conv2D, kernel_shape=3, nl=BNSeLU):
                with tf.variable_scope(name):
                    l = Conv2D('conv1_1', l, 32)
                    l = Conv2D('conv1_2', l, 32)
                    b1 = branch('branch1', l, 1, 1)
                    l = MaxPooling('pool1', l, 2)

                    l = Conv2D('conv2_1', l, 64)
                    l = Conv2D('conv2_2', l, 64)
                    b2 = branch('branch2', l, 1, 2)
                    l = MaxPooling('pool2', l, 2)

                    l = Conv2D('conv3_1', l, 128)
                    l = Conv2D('conv3_2', l, 128)
                    b3 = branch('branch3', l, 1, 4)
                    l = MaxPooling('pool3', l, 2) 
                    return [b1, b2, b3]

        def BSSloss(name, pred_list, gt, extra=True, weights=[0.5, 0.75, 1.0, 0.75, 1.0]):
            costs = []
            for idx, b in enumerate(pred_list):
                output = tf.nn.sigmoid(b, name='{}-output{}'.format(name,idx + 1))
                xentropy = class_balanced_sigmoid_cross_entropy(
                    b, gt, name='{}-xentropy{}'.format(name,idx + 1)) * weights[idx]
                costs.append(xentropy)
            if extra:
                costs.append(
                    tf.reduce_mean(
                        tf.square(
                            tf.nn.sigmoid(pred_list[-1]) - tf.cast(gt, tf.float32))))
            return costs

        segment_map = network1('segment', image)
        segment_map.append(Conv2D('segment-fuse',
                            tf.concat(segment_map, 3), 1, 1,
                            W_init=tf.constant_initializer(0.25),
                            use_bias=False, nl=tf.identity))
        segment_costs = BSSloss('segment', segment_map, segment)

        merge_data = tf.concat([image, segment_map[-1]], 3)

        skeleton_map = network2('skeleton', merge_data)
        skeleton_map.append(Conv2D('skeleton-fuse',
                            tf.concat(skeleton_map, 3), 1, 1,
                            W_init=tf.constant_initializer(0.33),
                            use_bias=False, nl=tf.identity))
        skeleton_costs = BSSloss('skeleton', 
            skeleton_map, skeleton, weights=[0.5, 0.75, 1.0, 1.0])


        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            segment_wd_cost = tf.multiply(wd_w, regularize_cost('segment/.*/W', 
                tf.nn.l2_loss), name='segment_wd_cost')
            segment_costs.append(segment_wd_cost)     
            segment_cost = tf.add_n(segment_costs, name='segment_cost')

            skeleton_wd_cost = tf.multiply(wd_w, regularize_cost('skeleton/.*/W', 
                tf.nn.l2_loss), name='skeleton_wd_cost')
            skeleton_costs.append(skeleton_wd_cost)     
            skeleton_cost = tf.add_n(skeleton_costs, name='skeleton_cost')

            add_moving_summary(segment_cost, skeleton_cost)

            loss = [segment_cost, skeleton_cost]
            
            self.cost = tf.add_n(loss, name='cost')
            add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-3, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('.*-fuse.*', 0.1), ('.*conv5_.*', 5)])])

def get_data(name):
    isTrain = name == 'train'
    ds = dataset.RoadNetImage2(name, 
        '/home/tensorflow/yhl/tensorpack_data/Guangliang/dataset', shuffle=True)
    print ds.size()
    class CropMultiple16(imgaug.ImageAugmentor):
        def _get_augment_params(self, img):
            newh = img.shape[0] // 16 * 16
            neww = img.shape[1] // 16 * 16
            assert newh > 0 and neww > 0
            diffh = img.shape[0] - newh
            h0 = 0 if diffh == 0 else self.rng.randint(diffh)
            diffw = img.shape[1] - neww
            w0 = 0 if diffw == 0 else self.rng.randint(diffw)
            return (h0, w0, newh, neww)

        def _augment(self, img, param):
            h0, w0, newh, neww = param
            return img[h0:h0 + newh, w0:w0 + neww]

    if isTrain:
        shape_aug = [
            #imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
            #                    aspect_ratio_thres=0.15),
            #imgaug.RotationAndCropValid(90),
            CropMultiple16(),
            imgaug.Flip(horiz=True),
            #imgaug.Flip(vert=True)
        ]
    else:
        # the original image shape (321x481) in BSDS is not a multiple of 16
        IMAGE_SHAPE = (512, 512)
        shape_aug = [imgaug.CenterCrop(IMAGE_SHAPE)]
    ds = AugmentImageComponents(ds, shape_aug, (0, 1, 2), copy=False)

    def f(m):   # thresholding
        #m = m / 255.0
        m[m >= 0.50] = 1
        m[m < 0.50] = 0
        return m
    ds = MapDataComponent(ds, f, 1)
    ds = MapDataComponent(ds, f, 2)

    if isTrain:
        augmentors = [
            imgaug.Brightness(63, clip=False),
            imgaug.Contrast((0.4, 1.5)),
        ]
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = BatchDataByShape(ds, 1, idx=0)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds

def view_data():
    ds = RepeatedData(get_data('train'), -1)
    ds.reset_state()
    for ims, edgemaps in ds.get_data():
        for im, edgemap in zip(ims, edgemaps):
            assert im.shape[0] % 16 == 0 and im.shape[1] % 16 == 0, im.shape
            cv2.imshow("im", im / 255.0)
            cv2.waitKey(1000)
            cv2.imshow("edge", edgemap)
            cv2.waitKey(1000)


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    steps_per_epoch = 400#dataset_train.size()
    #dataset_val = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', 
                [(20, 5e-4), (40, 1e-4), (60, 5e-5), (80, 1e-5), (100, 1e-6)]),
            HumanHyperParamSetter('learning_rate')
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=200,
    )


def run(model_path, image_path, output):
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['segment-output6', 'skeleton-output5'])
    predictor = OfflinePredictor(pred_config)

    imgs = glob.glob(os.path.join(image_path, '*.png'))
    for ls in imgs:
        im = cv2.imread(ls)
        assert im is not None
        im = cv2.resize(im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16))
        start = time.clock()
        outputs = predictor([[im.astype('float32')]])
        end = time.clock()
        time_consum.append(end-start)
        if output is None:
            for k in range(6):
                pred = outputs[k][0]
                cv2.imwrite("out{}.png".format(
                    '-fused' if k == 5 else str(k + 1)), pred * 255)
        else:
            fname = ls.split('/')[-1]
            fname = fname.split('.')[0]
            mask = cv2.merge([outputs[0][0], outputs[1][0], outputs[1][0]])
            cv2.imwrite(os.path.join(output,fname+'.png'), mask*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print 'image num: {}'.format(len(imgs))
    print 'average infer time (ms): %.3f'%(sum(time_consum)/len(time_consum)) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view_data()
    elif args.run:
        run(args.load, args.run, args.output)
    else:
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        SyncMultiGPUTrainer(config).train()
