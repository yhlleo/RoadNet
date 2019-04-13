import cv2
import tensorflow as tf
import numpy as np

sys.path.insert(0, '../../tensorpack')

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

# CEDN
def cedn(name, l):
    with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu), \
        argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
        with tf.variable_scope(name):
            l = Conv2D('conv1_1', l, 64)
            l = Conv2D('conv1_2', l, 64)
            l = MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            l = MaxPooling('pool4', l, 2)
            
            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            l = MaxPooling('pool5', l, 2)
            
            l = Conv2D('conv6_1', l, 2048, kernel_shape=7, padding='SAME')
            l = Dropout(l, 0.5)
            
            l = Conv2D('conv6_2', l, 512, kernel_shape=1)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up5', l, 512)
            l = Conv2D('deconv5', l, 512, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up4', l, 512)
            l = Conv2D('deconv4', l, 256, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up3', l, 256)
            l = Conv2D('deconv3', l, 128, kernel_shape=5)
            l = Dropout(l, 0.5)
                      
            l = Deconv2D('up2', l, 128)
            l = Conv2D('deconv2', l, 64, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up1', l, 64)
            l = Conv2D('deconv1', l, 32, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Conv2D('pred', l, 1, kernel_shape=5, nl=tf.identity)
        return l

def cednc(name, l, extra=True):
    with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu), \
        argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
        with tf.variable_scope(name):
            l = Conv2D('conv1_1', l, 32)
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
                
            l = Conv2D('conv5_1', l, 512, kernel_shape=7, padding='SAME')
            l = Dropout(l, 0.5)
            
            l = Conv2D('conv5_2', l, 512, kernel_shape=1)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up4', l, 512)
            l = Conv2D('deconv4', l, 256, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up3', l, 256)
            l = Conv2D('deconv3', l, 128, kernel_shape=5)
            l = Dropout(l, 0.5)
                      
            l = Deconv2D('up2', l, 128)
            l = Conv2D('deconv2', l, 64, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Deconv2D('up1', l, 64)
            l = Conv2D('deconv1', l, 32, kernel_shape=5)
            l = Dropout(l, 0.5)
            
            l = Conv2D('pred', l, 1, kernel_shape=5, nl=tf.identity)
        return l

# FCN 
def fcn_branch(name, x, y):
    with tf.variable_scope(name) as scope:
        return y + Conv2D('convfc', x, 1, kernel_shape=1, nl=tf.identity)
        
def fcn8s(name, l):
    with argscope(Conv2D, kernel_shape=3, nl=BNReLU), \
        argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
        with tf.variable_scope(name):
            l = Conv2D('conv1_1', l, 64)
            c1 = Conv2D('conv1_2', l, 64)
            l = MaxPooling('pool1', c1, 2)

            l = Conv2D('conv2_1', l, 128)
            c2 = Conv2D('conv2_2', l, 128)
            l = MaxPooling('pool2', c2, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            c3 = Conv2D('conv3_3', l, 256)
            p3 = MaxPooling('pool3', c3, 2)

            l = Conv2D('conv4_1', p3, 512)
            l = Conv2D('conv4_2', l, 512)
            c4 = Conv2D('conv4_3', l, 512)
            p4 = MaxPooling('pool4', c4, 2)
            
            l = Conv2D('conv5_1', p4, 512)
            l = Conv2D('conv5_2', l, 512)
            c5 = Conv2D('conv5_3', l, 512)
            p5 = MaxPooling('pool5', c5, 2)
            
            l = Conv2D('fc6', p5, 4096)
            l = Dropout(l, keep_prob=0.5)
            
            l = Conv2D('fc7', l, 4096)
            l = Dropout(l, keep_prob=0.5)
            
            l = Deconv2D('deconv1', l, 1)
            l = fcn_branch('sum1', l, p4)
            
            l = Deconv2D('deconv2', l, 1)
            l = fcn_branch('sum2', l, p3)
            
            l = Deconv2D('deconv3', l, 1, kerner_shape=16, stride=8)
            return l
            
def fcnc(name, l, extra=True):
    with argscope(Conv2D, kernel_shape=3, nl=BNReLU), \
        argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
        with tf.variable_scope(name):
            l = Conv2D('conv1_1', l, 64)
            c1 = Conv2D('conv1_2', l, 64)
            l = MaxPooling('pool1', c1, 2)

            l = Conv2D('conv2_1', l, 128)
            c2 = Conv2D('conv2_2', l, 128)
            p2 = MaxPooling('pool2', c2, 2)

            l = Conv2D('conv3_1', p2, 256)
            l = Conv2D('conv3_2', l, 256)
            c3 = Conv2D('conv3_3', l, 256)
            p3 = MaxPooling('pool3', c3, 2)

            l = Conv2D('conv4_1', p3, 512)
            l = Conv2D('conv4_2', l, 512)
            c4 = Conv2D('conv4_3', l, 512)
            p4 = MaxPooling('pool4', c4, 2)
            
            l = Conv2D('fc6', p4, 2048)
            l = Dropout(l, keep_prob=0.5)
            
            l = Conv2D('fc7', l, 2048)
            l = Dropout(l, keep_prob=0.5)
            
            l = Deconv2D('deconv1', l, 1)
            l = fcn_branch('sum1', l, p3)
            
            l = Deconv2D('deconv2', l, 1)
            l = fcn_branch('sum2', l, p2)
            
            l = Deconv2D('deconv3', l, 1, kerner_shape=8, stride=4)
            return l