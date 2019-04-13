#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: nonlin.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from .common import layer_register
from .batch_norm import BatchNorm

__all__ = ['Maxout', 'PReLU', 'LeakyReLU', 'SeLU', 'BNSeLU', 'Swish', 'BNReLU', 'BNELU', 'BNCReLU']


@layer_register(use_scope=False)
def Maxout(x, num_unit):
    """
    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
        num_unit (int): a int. Must be divisible by C.

    Returns:
        tf.Tensor: of shape NHW(C/num_unit) named ``output``.
    """
    input_shape = x.get_shape().as_list()
    ndim = len(input_shape)
    assert ndim == 4 or ndim == 2
    ch = input_shape[-1]
    assert ch is not None and ch % num_unit == 0
    if ndim == 4:
        x = tf.reshape(x, [-1, input_shape[1], input_shape[2], ch / num_unit, num_unit])
    else:
        x = tf.reshape(x, [-1, ch / num_unit, num_unit])
    return tf.reduce_max(x, ndim, name='output')


@layer_register(log_shape=False)
def PReLU(x, init=0.001, name='output'):
    """
    Parameterized ReLU as in the paper `Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification
    <http://arxiv.org/abs/1502.01852>`_.

    Args:
        x (tf.Tensor): input
        init (float): initial value for the learnable slope.
        name (str): name of the output.

    Variable Names:

    * ``alpha``: learnable slope.
    """
    init = tf.constant_initializer(init)
    alpha = tf.get_variable('alpha', [], initializer=init)
    x = ((1 + alpha) * x + (1 - alpha) * tf.abs(x))
    return tf.multiply(x, 0.5, name=name)


@layer_register(use_scope=False, log_shape=False)
def LeakyReLU(x, alpha, name='output'):
    """
    Leaky ReLU as in paper `Rectifier Nonlinearities Improve Neural Network Acoustic
    Models
    <http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_.

    Args:
        x (tf.Tensor): input
        alpha (float): the slope.
    """
    return tf.maximum(x, alpha * x, name=name)

@layer_register(use_scope=False, log_shape=False)
def SeLU(x, name="output"):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

@layer_register(use_scope=False, log_shape=False)
def BNSeLU(x, name=None):
    x = BatchNorm('bn', x)
    return SeLU(x, name=name)

@layer_register(use_scope=False, log_shape=False)
def Swish(x, name='output'):
    return x*tf.nn.sigmoid(x, name=name)

@layer_register(use_scope=False, log_shape=False)
def BNPReLU(x, init=0.001, name=None):
    """
    A shorthand of BatchNormalization + PReLU
    """
    x = BatchNorm('bn', x)
    return PReLU(x, init, name)
    
@layer_register(use_scope=False, log_shape=False)
def BNLReLU(x, alpha=0.01, name=None):
    """
    A shorthand of BatchNormalization + LeakyReLU
    """
    x = BatchNorm('bn', x)
    return LeakyReLU(x, alpha, name)


@layer_register(log_shape=False, use_scope=False)
def BNReLU(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.
    """
    x = BatchNorm('bn', x)
    x = tf.nn.relu(x, name=name)
    return x

@layer_register(use_scope=False, log_shape=False)
def BNELU(x, name=None):
    """
    A shorthand of BatchNormalization + ELU
    ELU as in paper `Fast and Accuracy Deep Network Learning by Exponential Linear Units
    <https://arxiv.org/pdf/1511.07289v5.pdf>`_.
    """
    x = BatchNorm('bn', x)
    return tf.nn.elu(x, name)

@layer_register(use_scope=False, log_shape=False)
def BNCReLU(x, name=None):
    """
    A shorthand of BatchNormalization + CReLU
    """
    x = BatchNorm('bn', x)
    return tf.nn.crelu(x, name)
