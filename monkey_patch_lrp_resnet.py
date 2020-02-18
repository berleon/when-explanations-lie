import tensorflow
import tensorflow as tf
import warnings

import innvestigate
import matplotlib.pyplot as plt

import numpy as np
import PIL
import copy
import contextlib

import imp
import numpy as np
import os

from skimage.measure import compare_ssim
import pickle
from collections import OrderedDict
from IPython.display import IFrame, display

import keras
import keras.backend
import keras.models


import innvestigate
import innvestigate.applications.imagenet
import innvestigate.utils as iutils
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRP
from innvestigate.analyzer.base import AnalyzerNetworkBase, ReverseAnalyzerBase
from innvestigate.analyzer.deeptaylor import DeepTaylor

import time
import tqdm

import seaborn as sns

import itertools
import matplotlib as mpl

from tensorflow.python.client import device_lib


import inspect
import keras
import keras.backend as K
import keras.engine.topology
import keras.models
import keras.layers
import keras.layers.convolutional
import keras.layers.core
import keras.layers.local
import keras.layers.noise
import keras.layers.normalization
import keras.layers.pooling


from innvestigate.analyzer import base
from innvestigate import layers as ilayers
from innvestigate import utils as iutils
import innvestigate.utils.keras as kutils
from innvestigate.utils.keras import checks as kchecks
from innvestigate.utils.keras import graph as kgraph
from innvestigate.analyzer.relevance_based import relevance_rule as rrule
from innvestigate.analyzer.relevance_based import utils as rutils
import innvestigate.analyzer.relevance_based.relevance_analyzer




def add_init(shape, dtype=None):
    # print(shape)
    h, w, cin, cout = shape
    weight = np.zeros((cin, cout))
    n_inputs = cin // cout

    weight = np.concatenate([np.eye(cout) for _ in range(n_inputs)])
    #print(weight)
    #plt.imshow(weight)
    #plt.show()
    return weight[None, None]


def get_add_reverse_layer_cls_with_rule(rule):
    class AddReverseLayerWithRule(kgraph.ReverseMappingBase):
        """Special Add layer handler that applies the Z-Rule"""

        def __init__(self, layer, state):
            #print("in AddReverseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
            self._layer_wo_act = kgraph.copy_layer_wo_activation(layer,
                                                                 name_template="reversed_kernel_%s")

            input_channels = [int(i.shape[-1]) for i in layer.input]
            self._merge_layer = keras.layers.Concatenate()


            self._sum_layer_with_kernel = keras.layers.Conv2D(input_channels[0], (1, 1),
                                                              #kernel_initializer=add_init,
                                                              use_bias=False)
            self._sum_layer_with_kernel.build((None, None, None, sum(input_channels)))
            #self._sum_layer_with_kernel.weights[0].initializer.run(session=K.get_session())

            weight_shape = [int(d) for d in self._sum_layer_with_kernel.weights[0].shape]
            self._sum_layer_with_kernel.set_weights([add_init(weight_shape)])

            x = self._merge_layer(layer.input)
            x = self._sum_layer_with_kernel(x)

            self._rule = rule(self._sum_layer_with_kernel, state)

        def apply(self, Xs, Ys, Rs, reverse_state):
            def slice_channels(start, end):
                def wrapper(x):
                    x_slice = x[:, :, :, start:end]
                    return K.clip(x_slice, 0, 1000)
                return wrapper
            merge_Xs = [self._merge_layer(Xs)]

            R_conv = self._rule.apply(merge_Xs, Ys, Rs, reverse_state)[0]
            # unmerge
            R_returns = []
            b, h, w, c = R_conv.shape
            cin = c // len(Xs)
            for i in range(len(Xs)):
                R_returns.append(keras.layers.Lambda(slice_channels(i*cin, (i+1)*cin))(R_conv))

            return [r for r in R_returns]

    return AddReverseLayerWithRule


def get_bn_reverse_layer_cls_with_rule(rule):
    class BatchNormalizationReverseWithRuleLayer(kgraph.ReverseMappingBase):
        """Special BN handler that applies the Z-Rule"""

        def __init__(self, layer, state):
            ##print("in BatchNormalizationReverseLayer.init:", layer.__class__.__name__,"-> Dedicated ReverseLayer class" ) #debug
            config = layer.get_config()

            self._center = config['center']
            self._scale = config['scale']
            self._axis = config['axis']

            self._mean = layer.moving_mean
            self._var = layer.moving_variance
            if self._center:
                self._beta = layer.beta
            else:
                self._beta = K.zeros_like(self._mean)
            if self._scale:
                self._gamma = layer.gamma
            else:
                self._gamma = K.ones_like(self._mean)


            channels = int(self._beta.shape[0])
            self._bn_as_conv_layer = keras.layers.DepthwiseConv2D((1, 1), use_bias=True)
            self._bn_as_conv_layer.build((None, None, None, channels))
            self._bn_as_conv_layer.weights[0].initializer.run(session=K.get_session())
            self._bn_as_conv_layer.weights[1].initializer.run(session=K.get_session())

            # `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`
            #         = x / var_eps * gamma  - gamma * mean / var_eps + beta
            #
            var_eps = tf.sqrt(self._var + config['epsilon'])
            bias = - self._gamma * self._mean / var_eps + self._beta
            kernel = self._gamma / var_eps

            self._bn_as_conv_layer.depthwise_kernel = tf.identity(kernel[None, None, :, None], name='bn_as_conv_layer_kernel')
            self._bn_as_conv_layer.bias = tf.identity(bias, name='bn_as_conv_layer_bias')
            self._bn_as_conv_layer._trainable_weights = []
            self._bn_as_conv_layer._non_trainable_weights = [self._bn_as_conv_layer.depthwise_kernel, self._bn_as_conv_layer.bias]

            x = self._bn_as_conv_layer(layer.input)

            self.rule = rule(self._bn_as_conv_layer, state)

        def apply(self, Xs, Ys, Rs, reverse_state):
            ##print("    in BatchNormalizationReverseLayer.apply:", reverse_state['layer'].__class__.__name__, '(nid: {})'.format(reverse_state['nid']))
            rs = self.rule.apply(Xs, Ys, Rs, reverse_state)
            if False:
                w, b = self._bn_as_conv_layer.get_weights()
                plt.title(w.shape)
                plt.imshow(w[0, 0])
                plt.show()
            return rs

    return BatchNormalizationReverseWithRuleLayer


@contextlib.contextmanager
def custom_add_bn_rule(rule):
    try:
        #
        old_add_cls = copy.deepcopy(innvestigate.analyzer.relevance_based.relevance_analyzer.AddReverseLayer)
        old_bn_cls = copy.deepcopy(innvestigate.analyzer.relevance_based.relevance_analyzer.BatchNormalizationReverseLayer)
        if rule is not None:
            add_cls = get_add_reverse_layer_cls_with_rule(rule)
            bn_cls = get_bn_reverse_layer_cls_with_rule(rule)
            print('monkey patching add reverse class with rule', rule)
            print('monkey patching bn reverse class with rule', rule)
            innvestigate.analyzer.relevance_based.relevance_analyzer.AddReverseLayer = add_cls
            innvestigate.analyzer.relevance_based.relevance_analyzer.BatchNormalizationReverseLayer = bn_cls
        yield
    finally:
        innvestigate.analyzer.relevance_based.relevance_analyzer.AddReverseLayer = old_add_cls
        innvestigate.analyzer.relevance_based.relevance_analyzer.BatchNormalizationReverseLayer = old_bn_cls


def alpha_beta_wrapper(alpha, beta):
    class AlphaBetaRuleWrapper(rrule.AlphaBetaRule):
        def __init__(self, layer, state, bias=True, copy_weights=False):
            super(AlphaBetaRuleWrapper, self).__init__(layer, state, alpha=alpha, beta=beta,
                             bias=bias, copy_weights=copy_weights)

        def __repr__(self):
            return "AlphaBetaRuleWrapper(alpha={}, beta={})".format(self._alpha, self._beta)

    return AlphaBetaRuleWrapper

def get_custom_rule(innv_name, kwargs):
    if innv_name == 'lrp.alpha_beta':
        return alpha_beta_wrapper(kwargs['alpha'], kwargs['beta'])
    elif innv_name == 'lrp.alpha_1_beta_0':
        return alpha_beta_wrapper(1, 0)
    elif innv_name == 'lrp.alpha_2_beta_1':
        return alpha_beta_wrapper(2, 1)
    elif innv_name == 'lrp.sequential_preset_a':
        return alpha_beta_wrapper(1, 0)
    elif innv_name == 'lrp.sequential_preset_b':
        return alpha_beta_wrapper(2, 1)



def _reverse_model(self,
                   model,
                   stop_analysis_at_tensors=[],
                   return_all_reversed_tensors=False):
    ret = kgraph.reverse_model(
        model,
        reverse_mappings=self._reverse_mapping,
        default_reverse_mapping=self._default_reverse_mapping,
        head_mapping=self._head_mapping,
        stop_mapping_at_tensors=stop_analysis_at_tensors,
        verbose=self._reverse_verbose,
        clip_all_reversed_tensors=self._reverse_clip_values,
        project_bottleneck_tensors=self._reverse_project_bottleneck_layers,
        return_all_reversed_tensors=return_all_reversed_tensors)

    if return_all_reversed_tensors:
        self._reversed_tensors_raw = ret[1]
    return ret

def _prepare_model(self, model):
    return super(DeepTaylor, self)._prepare_model(model)

# otherwise DTD does not work on negative outputs
def apply_static_monkey_patches():
    innvestigate.analyzer.base.ReverseAnalyzerBase._reverse_model = _reverse_model
    DeepTaylor._prepare_model = _prepare_model


apply_static_monkey_patches()