# helper functions 
import glob 
import tensorflow
import tensorflow as tf

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


def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    if ret.ndim == 2:
        ret.resize((size, size, 1))
        ret = np.repeat(ret, 3, axis=-1)
    return ret

def preprocess(X, net):
    X = X.copy()
    X = net["preprocess_f"](X)
    return X

def image(X):
    X = X.copy()
    return ivis.project(X, absmax=255.0)

def to_heatmap(saliency):
    return ivis.heatmap(
        np.abs(
            ivis.clip_quantile(saliency.copy(), 0.5))
        )

def copy_weights(model_to, model_from, idxs):
    idxs = set(idxs)
    all_weights = []
    for i in range(len(model_to.layers)):
        if i in idxs:
            all_weights.extend(model_from.layers[i].get_weights())
        else:
            all_weights.extend(model_to.layers[i].get_weights())
    model_to.set_weights(all_weights)
        
def get_random_target(target_exclude):
    return np.random.choice([i for i in range(1000) if i != target_exclude])

def normalize(x, percentile=99):
    """
    all heatmap are normalized
    """
    vmin = np.percentile(x, percentile)
    vmax = np.percentile(x, 100 - percentile)
    return np.clip((x - vmin) / (vmax - vmin), 0, 1)

def ssim_flipped(x, y, win_size=5, **kwargs):
    norm_x = normalize(x)
    norm_y = normalize(y)
    norm_y_flip = normalize(-y)
    
    ssim = compare_ssim(norm_x, norm_y, win_size=win_size, **kwargs)
    ssim_flip = compare_ssim(norm_x, norm_y_flip, win_size=win_size, **kwargs)
    return max(ssim, ssim_flip)


def l2_flipped(x, y):
    norm_x = normalize(x)
    norm_y = normalize(y)
    norm_y_flip = normalize(-y)
    l2 = np.mean(np.sqrt((norm_x - norm_y)**2))
    l2_flipped = np.mean(np.sqrt((norm_x - norm_y_flip)**2))
    return max(l2, l2_flipped)


def cosine_similarity(U, V):
    v_norm =  V / np.linalg.norm(V, axis=0, keepdims=True)
    u_norm = U / np.linalg.norm(U, axis=0, keepdims=True)
    return (v_norm.T @ u_norm)


def norm_image(x):
    mi = x.min()
    ma = x.max()
    return (x - mi) / (ma - mi)

def load_image_paths(validation_dir):
    val_filepaths = glob.glob(validation_dir + '/**/*.JPEG')
    val_filepaths = sorted(val_filepaths)
    val_targets = glob.glob(validation_dir + '/*')
    val_targets = [t.split('/')[-1] for t in sorted(val_targets)]
    val_path_with_target = []
    for path in val_filepaths:
        synnet = path.split('/')[-2]
        target = val_targets.index(synnet)
        val_path_with_target.append((path, target))
    return val_path_with_target

def load_val_images(innv_net, imagenet_val_dir, ex_image_path, n_selected_imgs):
    val_paths = load_image_paths(imagenet_val_dir)
    ex_image_full_path, ex_target = [
        (path, target) for (path, target) in val_paths 
        if path.endswith(ex_image_path)][0]
    
    ex_image_idx = val_paths.index((ex_image_full_path, ex_target))

    np.random.seed(0)
    selected_img_idxs = [ex_image_idx] + np.random.choice([idx for idx in range(len(val_paths))
                                                           if idx != ex_image_idx], n_selected_imgs - 1).tolist()
    
    
    ex_image = preprocess(load_image(os.path.join(imagenet_val_dir, ex_image_path), 224),
                          innv_net)[np.newaxis]
    ex_target = [target for (path, target) in val_paths 
                 if path.endswith(ex_image_path)][0]

    val_images = [(
        preprocess(
            load_image(val_paths[idx][0], 224),
            innv_net
        )[np.newaxis],
        val_paths[idx][1]) for idx in selected_img_idxs]
    
    return ex_image, ex_target, val_images, selected_img_idxs


def load_model(model='vgg16', load_weights=True, load_patterns="relu"):
    load_func = getattr(innvestigate.applications.imagenet, model)
    net = load_func(load_weights=load_weights, load_patterns=load_patterns)
    model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    
    channels_first = keras.backend.image_data_format() == "channels_first"
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
    return model_wo_softmax, net, color_conversion

def get_output_shapes(model, input_shape=(None, 224, 224, 3)):
    model_output_shapes = OrderedDict()
    for i, layer in enumerate(model.layers):
        model_output_shapes[i] = layer.get_output_shape_at(0)
    return model_output_shapes

def get_nice_resnet_layers(model):
    def get_conv_name(layer):
        block, branch = layer.name.split('_')
        block_spec = block.lstrip('res')
        block_idx = block_spec[0]
        subblock_idx = ' abcdefg'.index(block_spec[1])
        layer_idx = branch[-1]
        if layer_idx == '1':
            layer_idx = 's'
            nice_name = "conv{}_skip".format(block_idx)
        else:
            nice_name = "conv{}_{}{}".format(block_idx, subblock_idx, layer_idx) 
        return nice_name
        
    nice_layer_names_resnet = OrderedDict([(0, "input")])
    for i in range(len(model.layers)):
        layer = model.get_layer(index=i)
        if type(layer) == keras.layers.Conv2D:
            if layer.name == 'conv1':
                nice_layer_names_resnet[i] = 'conv1'
                continue

            nice_layer_names_resnet[i] = get_conv_name(layer)
            #print(block_spec, block_idx, subblock_idx, layer.name, nice_name)
        elif type(layer) == keras.layers.MaxPool2D:
            nice_layer_names_resnet[i] = 'maxpool'
        elif type(layer) == keras.layers.GlobalAveragePooling2D:
            nice_layer_names_resnet[i] = 'avgpool'
        elif type(layer) == keras.layers.Add:
            prev_layer = model.get_layer(index=i-1)
            print(prev_layer)
            block, branch = prev_layer.name.split('_')
            block_spec = block.lstrip('bn')
            block_idx = block_spec[0]
            subblock_idx = ' abcdefg'.index(block_spec[1])
            nice_name = 'block{}_{}'.format(block_idx, subblock_idx)
            # print(prev_layer.name, nice_name, block_idx, subblock_idx, 
            #       model.get_layer(index=i+1).name)
            # print(nice_name)
            # name the following activation layers
            nice_layer_names_resnet[i+1] = nice_name
        elif type(layer) == keras.layers.Dense:
            nice_layer_names_resnet[i] = 'dense'
    return nice_layer_names_resnet


def get_layer_idx_full(model_name, nice_layer_names, layer_name):
    for key, l in nice_layer_names[model_name].items():
        if l == layer_name:
            return key
        
    raise ValueError("no layer: " + layer_name)
    
def get_nice_layer_names(resnet):
    return {
        'vgg16': OrderedDict([
            (0, 'input'),
            (1, 'conv1_1'),
            (2, 'conv1_2'),
            (3, 'pool1'),
            (4, 'conv2_1'),
            (5, 'conv2_2'),
            (6, 'pool2'),
            (7, 'conv3_1'),
            (8, 'conv3_2'),
            (9, 'conv3_3'),
            (10, 'pool3'),
            (11, 'conv4_1'),
            (12, 'conv4_2'),
            (13, 'conv4_3'),
            (14, 'pool4'),
            (15, 'conv5_1'),
            (16, 'conv5_2'),
            (17, 'conv5_3'),
            (18, 'pool5'),
            (19, 'flatten'),
            (20, 'fc1'),
            (21, 'fc2'),
            (22, 'fc3') 
        ]),
        'resnet50': get_nice_resnet_layers(resnet)
    }

def heatmap_postprocess(name, x):
    if name == 'abs.max':
        return np.abs(x).max(-1)
    elif name == 'abs.sum':
        return np.abs(x).sum(-1)
    elif name == 'sum':
        return x.sum(-1)
    else:
        pass
    
def get_analyser_params(input_range, smoothgrad_scale=0.15):
    noise_scale = smoothgrad_scale * (input_range[1] - input_range[0])
    return [
        ('GuidedBP',  "guided_backprop",  'abs.max', [], {}),
        ('Deconv',  "deconvnet",  'abs.max', [], {}),
        ('DTD', "deep_taylor.bounded", 'sum', [],
             {"low": input_range[0], "high": input_range[1]}),
        ('LRP $\\alpha1\\beta0$',  'lrp.alpha_beta', 
             'sum', [], {'alpha': 1, 'beta': 0}),
        ('LRP $\\alpha2\\beta1$',  'lrp.alpha_beta', 
             'sum', [], {'alpha': 2, 'beta': 1}),
        ('LRP $\\alpha5\\beta4$',  'lrp.alpha_beta', 
             'sum', [], {'alpha': 5, 'beta': 4}),
        #('$\\alpha=10, \\beta=9$-LRP',  'lrp.alpha_beta', 
        #     'sum', [], {'alpha': 10, 'beta': 9}),
        ('LRP CMP $\\alpha1\\beta0$', 'lrp.sequential_preset_a', 
             'sum', [], {"epsilon": 1e-10}), 
        ('LRP CMP $\\alpha2\\beta1$',  'lrp.sequential_preset_b', 
             'sum', [], {"epsilon": 1e-10}), 
        ("PatternAttr.", "pattern.attribution",
             'sum', ['exclude_resnet50'], {}),
        ("DeepLift R.Can.", "deeplift.wrapper", 'sum', [],
             {'nonlinear_mode': 'reveal_cancel'}),
        ("DeepLift Resc.",  "deeplift.wrapper", 'sum', [],
             {'nonlinear_mode': 'rescale'}),
        ('LRP-z', 'lrp.z', 'sum', [], {}),
        ('SmoothGrad',  "smoothgrad",  'abs.max', ['exclude_cos_sim'],
             {"augment_by_n": 50, "noise_scale": noise_scale}),
        ('Gradient',  "gradient",  'abs.max', [], {}),
    ]
    
colors = sns.color_palette('colorblind', n_colors=5)

mpl_styles = OrderedDict([
    ('GuidedBP',                   {'marker': '$G$', 'color': colors[0]}),
    ('Deconv',                     {'marker': '$V$', 'color': colors[1]}),
    ('LRP-z',                      {'marker': 'D',   'color': colors[2]}),
    ('DTD',                        {'marker': '$T$', 'color': colors[3]}),
    ('PatternAttr.',               {'marker': '$P$', 'color': colors[4]}),
    ('LRP $\\alpha1\\beta0$',      {'marker': '<',   'color': colors[0]}),
    ('LRP $\\alpha2\\beta1$',      {'marker': '>',   'color': colors[1]}),
    ('LRP $\\alpha5\\beta4$',      {'marker': '^',   'color': colors[2]}),
    ('LRP CMP $\\alpha1\\beta0$',  {'marker': 's',   'color': colors[3]}),
    ('LRP CMP $\\alpha2\\beta1$',  {'marker': 'P',   'color': colors[4]}),
    ('DeepLift R.Can.',            {'marker': '$D$',   'color': colors[0]}),
    ('DeepLift Resc.',             {'marker': '$D$',   'color': colors[1]}),
    ('SmoothGrad',                 {'marker': 'o',   'color': colors[2]}),
    ('Gradient',                   {'marker': 'v',   'color': 'black'}),
])



def create_replacement_class(analyser_cls):
    assert issubclass(analyser_cls, ReverseAnalyzerBase)
    class ReplaceBackward(analyser_cls):
        def __init__(self, model, *args, **kwargs):
            kwargs['reverse_keep_tensors'] = True
            super().__init__(model, *args, **kwargs)
        
        def _create_analysis(self, *args, **kwargs):
            outputs, relevances_per_layer = super()._create_analysis(*args, **kwargs)
            self._relevances_per_layer = relevances_per_layer[::-1]
            return outputs, relevances_per_layer
        
        def _get_layer_idx(self, name):
            layer = self._model.get_layer(name='dense_2')
            return self._model.layers.index(layer)
        
        def get_relevances(self, input_value, relevance_value,  
                           set_layer, selected_layers):
            """
            return relevance values
            """
            sess = keras.backend.get_session()
            inp = self._analyzer_model.inputs[0]
            set_layer_idx = self._get_layer_idx(set_layer)
            selected_layer_idxs = [
                self._get_layer_idx(n) for n in selected_layers]
            rel_tensor = self._relevances_per_layer[set_layer_idx]
            
            return sess.run(
                [self._relevances_per_layer[i] for i in selected_layer_idxs],
                feed_dict={ 
                    inp: input_value,
                    rel_tensor: relevance_value
           })
        
    return ReplaceBackward 


def get_replacement_analyser(model, analyser_cls, **kwargs):
    if type(analyser_cls) == str:
        analyser_cls = innvestigate.analyzer.analyzers[analyser_cls]
    replacement_cls = create_replacement_class(analyser_cls)
    
    return replacement_cls(model, **kwargs)



def conv_as_matrix(x):
    if len(x.shape) == 2:
        return x
    if len(x.shape) == 3:
        x = x[None]
    b, h, w, c = x.shape
    return np.reshape(x, (b*h*w, c))


def cosine_similarity_dot(U, V):
    v_norm =  V / np.linalg.norm(V, axis=1, keepdims=True)
    u_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
    return (v_norm * u_norm).sum(1)

def pairwise_cosine_similarity(matrices):
    cos_sims = []
    for i, j in itertools.combinations(range(len(matrices)), 2):
        u = matrices[i]
        v = matrices[j]
        cos_sims.append(cosine_similarity_dot(u, v))
    return np.stack(cos_sims)

def cosine_similarities_from_relevances(relevance_per_layers):
    cos_sims = []
    for rel_per_layer in relevance_per_layers:
        rel_per_layer = [conv_as_matrix(r[None]) for r in rel_per_layer]

        cos_sims.append(pairwise_cosine_similarity(rel_per_layer).flatten())
    return cos_sims

def cosine_similarities_from_relevances(relevance_per_layers):
    cos_sims = []
    for rel_per_layer in relevance_per_layers:
        rel_per_layer = [conv_as_matrix(r[None]) for r in rel_per_layer]

        cos_sims.append(pairwise_cosine_similarity(rel_per_layer).flatten())
    return cos_sims