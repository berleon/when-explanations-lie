import numpy as np
import PIL 
import copy

import contextlib
import deeplift

from deeplift.layers import Dense, Concat, Merge
from deeplift.layers.core import DenseMxtsMode, SingleInputMixin, Node, NoOp
from deeplift.conversion.kerasapi_conversion import KerasKeys
import keras.backend as K
from keras.utils.generic_utils import transpose_shape
import tensorflow as tf
import keras

from deeplift.layers import helper_functions as hf
import tensorflow as tf
from deeplift.util import to_tf_variable
from deeplift.layers.convolutional import *


def monkey_patch_Merge_compute_shape(self, input_shapes):                       
    shape = []                                                                  
    # don't recompute input shapes
    # input_shapes = [an_input.get_shape() for an_input in self.inputs]         
    assert len(set(len(x) for x in input_shapes)) == 1, (
      "all inputs should have the same num"+                                    
      " of dims - got: "+str(input_shapes))          
    for dim_idx in range(len(input_shapes[0])):    
        lengths_for_that_dim = [input_shape[dim_idx]
                                for input_shape in input_shapes]
        if (dim_idx != self.axis):
            assert len(set(lengths_for_that_dim))==1, (
                   "lengths for dim "+str(dim_idx)  
                   +" should be the same, got: "+str(lengths_for_that_dim))
            shape.append(lengths_for_that_dim[0])
        else:
            shape.append(self.compute_shape_for_merge_axis(                     
                               lengths_for_that_dim))                           
    return shape     
Merge._compute_shape = monkey_patch_Merge_compute_shape



# copied from: https://github.com/kundajelab/deeplift/pull/76/files

class GlobalAvgPool2D(SingleInputMixin, Node):

    def __init__(self, **kwargs):
        super(GlobalAvgPool2D, self).__init__(**kwargs)

    def _compute_shape(self, input_shape):
        assert len(input_shape)==4
        shape_to_return = [None, input_shape[-1]]
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return tf.reduce_mean(input_act_vars, axis=(1,2))

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs)
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):
        height = self._get_input_activation_vars().get_shape().as_list()[1]
        width = self._get_input_activation_vars().get_shape().as_list()[2]
        mask = (tf.ones_like(self._get_input_activation_vars())/
                float(width*height))
        return tf.multiply(out_grad[:, None, None, :], mask)

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments

    
def globalavgpooling2d_conversion(config, name, verbose, **kwargs):
    return [layers.GlobalAvgPool2D(name=name, verbose=verbose)]

    
class ZeroPadding2D(NoOp):
    def __init__(self, padding, data_format=None, **kwargs):                
        # self.rank is 1 for ZeroPadding1D, 2 for ZeroPadding2D.            
        self.rank = len(padding)                                            
        self.padding = padding                                              
        self.data_format = K.normalize_data_format(data_format)             
        super(ZeroPadding2D, self).__init__(**kwargs)                       
                                                                            
    def _compute_shape(self, input_shape):                                  
        padding_all_dims = ((0, 0),) + tuple(self.padding) + ((0, 0),)
        spatial_axes = list(range(1, 1 + self.rank))           
        padding_all_dims = transpose_shape(padding_all_dims,    
                                           self.data_format,    
                                           spatial_axes)        
        output_shape = list(input_shape)                       
        for dim in range(len(output_shape)):                    
            if output_shape[dim] is not None:                   
                output_shape[dim] += sum(padding_all_dims[dim]) 
        return tuple(output_shape)                                          
                                                                            
    def _build_activation_vars(self, inputs):                               
        return K.spatial_2d_padding(inputs,                                 
                                    padding=self.padding,                   
                                    data_format=self.data_format) 

    def _build_pos_and_neg_contribs(self):
        input_pos_contribs, input_neg_contribs = self._get_input_pos_and_neg_contribs()
        return (self._build_activation_vars(input_pos_contribs), 
                self._build_activation_vars(input_neg_contribs))
        
    def _grad_op(self, out_grad):
        b, h, w, c = self._get_input_activation_vars().get_shape().as_list()
        (ha, hb), (wa, wb) = self.padding
        return out_grad[:, ha:ha+h, wa:wa+w]
    
    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments

    
def globalavgpooling2d_conversion(config, name, verbose, **kwargs):
    return [GlobalAvgPool2D(
             name=name,
             verbose=verbose)]


def zeropadding2d_conversion(config, name, verbose, **kwargs):                         
    padding = config[KerasKeys.padding]                                                
    data_format = config[KerasKeys.data_format]                                        
    return [ZeroPadding2D(padding, data_format, name=name)]  


class Add(Dense):                                                  
    """                                                            
    Is like a linear layer: [1 1] * [a b]^T                        
    """                                                            
    def __init__(self, **kwargs):         
        super(Add, self).__init__(                                 
            kernel=np.ones((2, 1)),                               
            bias=np.zeros((1,)),                                   
            dense_mxts_mode=DenseMxtsMode.Linear, **kwargs)        

        
class ConcatForAdd(Concat):
    def __init__(self, **kwargs):
        super().__init__(axis=4, **kwargs)
        
    def _build_activation_vars(self, input_act_vars):
        return super()._build_activation_vars([v[:, :, :, :, None] for v in input_act_vars])
 
    def _compute_shape(self, input_shape):
        input_shape_pad = [list(inp) + [1,] for inp in input_shape]
        output_shape = super()._compute_shape(input_shape_pad)
        return output_shape
    
    def _get_mxts_increments_for_inputs(self):
        pos, neg = super()._get_mxts_increments_for_inputs()
        return [p[:, :, :, :, 0] for p in pos], [n[:, :, :, :, 0] for n in neg]

    def _get_mxts_increments_for_inputs(self):                                  
        return (
            [self.get_pos_mxts()[:, :, :, :, i] for i in range(len(self.inputs))],
            [self.get_neg_mxts()[:, :, :, :, i] for i in range(len(self.inputs))],
        )
    
    
class FlattenForAdd(NoOp):
    def _compute_shape(self, input_shape):                         
        b, h, w, c, n = input_shape
        self.shape_tuple = (None, h, w, c)
        return (None, n)                                     
    
    def _build_activation_vars(self, input_act_vars):              
        shape = input_act_vars.shape                               
        self.shape_tensor = tf.shape(input_act_vars)
        return tf.reshape(input_act_vars, (-1, shape[-1]))

    def _get_mxts_increments_for_inputs(self):              
        input_shape = tf.shape(self.inputs._activation_vars)
        
        pos_mxts_increments = tf.reshape(self.get_pos_mxts(), input_shape)
        neg_mxts_increments = tf.reshape(self.get_neg_mxts(), input_shape)             
        return pos_mxts_increments, neg_mxts_increments       
    
class BackToConvForAdd(NoOp):
    def __init__(self, corresponding_flatten, **kwargs):
        self.corresponding_flatten = corresponding_flatten
        super().__init__(**kwargs)
        
    def _compute_shape(self, input_shape):
        return self.corresponding_flatten.shape_tuple
    
    def _build_activation_vars(self, input_act_vars):
        
        out = tf.reshape(input_act_vars, self.corresponding_flatten.shape_tensor[:-1])
        return out
    
    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs = self._get_input_pos_and_neg_contribs()                  
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs)
        return pos_contribs, neg_contribs                           
                                                                    
    def _unflatten_keeping_first(self, mxts):                       
        input_act_vars = self._get_input_activation_vars()          
        return tf.reshape(tensor=mxts,                              
                          shape=tf.shape(input_act_vars))           
                                                                    
    def _get_mxts_increments_for_inputs(self):              
        input_shape = tf.shape(self.inputs._activation_vars)
        pos_mxts_increments = tf.reshape(self.get_pos_mxts(), (-1, 1))
        neg_mxts_increments = tf.reshape(self.get_neg_mxts(), (-1, 1))             
        return pos_mxts_increments, neg_mxts_increments       
        
        
def add_conversion(config, name, verbose, **kwargs):               
    flatten = FlattenForAdd(name=name + '_flatten_add') 
    return deeplift.util.connect_list_of_layers([
        ConcatForAdd(name=name + "_cat_add"), flatten, 
        Add(name=name), BackToConvForAdd(flatten, name=name + '_back_to_conv')])

import importlib
import deeplift.conversion.kerasapi_conversion
from deeplift.conversion.kerasapi_conversion import layer_name_to_conversion_function as deeplift_layer_name_to_conversion_function

importlib.reload(deeplift.conversion.kerasapi_conversion)

copy_layer_name_to_conversion_function = copy.deepcopy(
    deeplift.conversion.kerasapi_conversion.layer_name_to_conversion_function)
    

def monkey_patched_layer_name_to_conversion_function(layer_name):  
    try: 
        return copy_layer_name_to_conversion_function(layer_name)
    except KeyError:
        return {
           'zeropadding2d': zeropadding2d_conversion,                                                                                                                                                                                                                              
           'globalaveragepooling2d': globalavgpooling2d_conversion,                                                                                                                                                                                                                              
           'add': add_conversion,                 
       }[layer_name.lower()]
    
deeplift.conversion.kerasapi_conversion.layer_name_to_conversion_function = \
    monkey_patched_layer_name_to_conversion_function


def deeplift_prediction(dp_model, output_layer=None):
    inp_layer_name = dp_model._input_layer_names[0]
    inp_layer = dp_model._name_to_layer[inp_layer_name]
    if output_layer is None:
        output_layer = list(dp_model._name_to_layer.values())[-1]
    
    outputs = output_layer.get_activation_vars()
    def wrapper(inp_val, ref_val):
        sess = K.get_session()
        return sess.run([outputs], feed_dict={
            inp_layer.get_activation_vars(): inp_val,
            inp_layer.get_reference_vars(): ref_val,
        })[0]
    return wrapper

    
class DeepLiftRelevanceReplacer:
    def __init__(self, deeplift_wrapper):
        self.deeplift_wrapper = deeplift_wrapper
        if not hasattr(self.deeplift_wrapper, "_deep_lift_func"): 
            self.deeplift_wrapper._create_deep_lift_func()
        self.model = self.deeplift_wrapper._deeplift_model
        self.layers = list(self.model._name_to_layer.values())
        self.layer_names = list(self.model._name_to_layer.keys())
        self.input_layer = self.layers[0]
       
    def _get_layer_idx(self, name):
        deeplift_name = name + '_0'
        layer_names = list(self.model._name_to_layer.keys())
        return layer_names.index(deeplift_name)
    
    def get_relevances(self, input_value,  relevance_value,
                       set_layer, selected_layers, reference=None):
        def run_single(single_image, single_relevance_value, single_reference):
            sess = keras.backend.get_session()
            return sess.run(
                [self.layers[idx]._target_contrib_vars 
                 for idx in selected_layer_idxs], 
                feed_dict={
                    self.input_layer.get_activation_vars(): single_image,
                    self.input_layer.get_reference_vars(): single_reference,
                    changed_layer._pos_mxts: single_relevance_value,
                    changed_layer._neg_mxts: single_relevance_value, 
                })
            
        set_layer_idx = self._get_layer_idx(set_layer)
        changed_layer = self.layers[set_layer_idx]
        selected_layer_idxs = [self._get_layer_idx(name) for name in selected_layers]
        
        if reference is None:
            reference = np.zeros_like(input_value)
            
        self.layers[-1].set_active()
        
        aggregated_contribs = [[] for _ in selected_layer_idxs]
        for i in range(len(input_value)):
            contribs = run_single(
                input_value[i:i+1],
                relevance_value[i:i+1],
                reference[i:i+1],
            )
            for i, cont in enumerate(contribs):
                aggregated_contribs[i].append(cont)
                
        self.layers[-1].set_inactive()
        
        return [np.concatenate(contrib) for contrib in aggregated_contribs]
    
    

_pos_to_pos_mxts = True
_neg_to_neg_mxts = True
_neg_to_pos_mxts = True
_pos_to_neg_mxts = True
_zero_mxts = True


class DenseMonkeyPatch:
    def _get_mxts_increments_for_inputs(self):
        if (self.dense_mxts_mode == DenseMxtsMode.Linear): 
            #different inputs will inherit multipliers differently according
            #to the sign of inp_diff_ref (as this sign was used to determine
            #the pos_contribs and neg_contribs; there was no breakdown
            #by the pos/neg contribs of the input)
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)

            kernel_T = tf.transpose(self.kernel)
            
            inp_mxts_increments = 0
            if _pos_to_pos_mxts:
                 inp_mxts_increments += pos_inp_mask*(
                    tf.matmul(self.get_pos_mxts(),
                              kernel_T*(hf.gt_mask(kernel_T, 0.0))))
                     
            if _neg_to_pos_mxts:
                 inp_mxts_increments += pos_inp_mask*(
                    tf.matmul(self.get_neg_mxts(),
                                kernel_T*(hf.lt_mask(kernel_T, 0.0))))
                    
            if _pos_to_neg_mxts:
                inp_mxts_increments += neg_inp_mask*(
                    tf.matmul(self.get_pos_mxts(),
                              kernel_T*(hf.lt_mask(kernel_T, 0.0))))
            if _neg_to_neg_mxts:
                inp_mxts_increments += neg_inp_mask*(
                    tf.matmul(self.get_neg_mxts(),
                                kernel_T*(hf.gt_mask(kernel_T, 0.0))))
            
            if _zero_mxts:
                inp_mxts_increments += zero_inp_mask*(
                    tf.matmul(0.5*(self.get_pos_mxts()
                                   +self.get_neg_mxts()), kernel_T))
            #pos_mxts and neg_mxts in the input get the same multiplier
            #because the breakdown between pos and neg wasn't used to
            #compute pos_contribs and neg_contribs in the forward pass
            #(it was based entirely on inp_diff_ref)
            return inp_mxts_increments, inp_mxts_increments
        else:
            raise RuntimeError("Unsupported mxts mode: "
                               +str(self.dense_mxts_mode))
            
class Conv2dMonkeyPatch:
    def _get_mxts_increments_for_inputs(self):
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        inp_diff_ref = self._get_input_diff_from_reference_vars()
        inp_act_vars = self.inputs.get_activation_vars()
        strides_to_supply = [1]+list(self.strides)+[1]

        if (self.data_format == DataFormat.channels_first):
            pos_mxts = tf.transpose(a=pos_mxts, perm=(0,2,3,1))
            neg_mxts = tf.transpose(a=neg_mxts, perm=(0,2,3,1))
            inp_diff_ref = tf.transpose(a=inp_diff_ref, perm=(0,2,3,1))
            inp_act_vars = tf.transpose(a=inp_act_vars, perm=(0,2,3,1))

        output_shape = tf.shape(inp_act_vars)

        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref, 0.0)

            inp_mxts_increments = 0
            
            if _pos_to_pos_mxts:
                inp_mxts_increments += pos_inp_mask*(
                            tf.nn.conv2d_transpose(
                                value=pos_mxts,
                                filter=self.kernel*hf.gt_mask(self.kernel, 0.0),
                                output_shape=output_shape,
                                padding=self.padding,
                                strides=strides_to_supply
                            ))
            if _neg_to_pos_mxts:
                inp_mxts_increments += pos_inp_mask*(
                           tf.nn.conv2d_transpose(
                                value=neg_mxts,
                                filter=self.kernel*hf.lt_mask(self.kernel, 0.0),
                                output_shape=output_shape,
                                padding=self.padding,
                                strides=strides_to_supply
                            ))
            if _pos_to_neg_mxts:
                inp_mxts_increments += neg_inp_mask*(
                            tf.nn.conv2d_transpose(
                                value=pos_mxts,
                                filter=self.kernel*hf.lt_mask(self.kernel, 0.0),
                                output_shape=output_shape,
                                padding=self.padding,
                                strides=strides_to_supply
                            ))
            if _neg_to_neg_mxts:
                inp_mxts_increments += neg_inp_mask*(
                           tf.nn.conv2d_transpose(
                                value=neg_mxts,
                                filter=self.kernel*hf.gt_mask(self.kernel, 0.0),
                                output_shape=output_shape,
                                padding=self.padding,
                                strides=strides_to_supply
                           ))
            if _zero_mxts:
                inp_mxts_increments += zero_inp_mask*tf.nn.conv2d_transpose(
                                value=0.5*(pos_mxts+neg_mxts),
                                filter=self.kernel,
                                output_shape=output_shape,
                                padding=self.padding,
                                strides=strides_to_supply)
            pos_mxts_increments = inp_mxts_increments
            neg_mxts_increments = inp_mxts_increments
        else:
            raise RuntimeError("Unsupported conv mxts mode: "
                               +str(self.conv_mxts_mode))

        if (self.data_format == DataFormat.channels_first):
            pos_mxts_increments = tf.transpose(a=pos_mxts_increments,
                                               perm=(0,3,1,2))
            neg_mxts_increments = tf.transpose(a=neg_mxts_increments,
                                               perm=(0,3,1,2))

        return pos_mxts_increments, neg_mxts_increments


@contextlib.contextmanager
def monkey_patch_deeplift_neg_pos_mxts(cross_mxts=True):
    global _pos_to_pos_mxts, _neg_to_neg_mxts 
    global _neg_to_pos_mxts, _pos_to_neg_mxts, _zero_mxts 
    
    _pos_to_pos_mxts = True
    _neg_to_neg_mxts = True
    _neg_to_pos_mxts = True
    _pos_to_neg_mxts = True
    _zero_mxts = True
    
    if cross_mxts == False:
        _neg_to_pos_mxts = False
        _pos_to_neg_mxts = False
    
    saved_dense_func = copy.deepcopy(
        deeplift.layers.core.Dense._get_mxts_increments_for_inputs)
    saved_conv_func = copy.deepcopy(
        deeplift.layers.convolutional.Conv2D._get_mxts_increments_for_inputs)
    
    deeplift.layers.core.Dense._get_mxts_increments_for_inputs = \
        DenseMonkeyPatch._get_mxts_increments_for_inputs

    deeplift.layers.convolutional.Conv2D._get_mxts_increments_for_inputs = \
        Conv2dMonkeyPatch._get_mxts_increments_for_inputs
    try:
        yield
    finally:
        deeplift.layers.core.Dense._get_mxts_increments_for_inputs = saved_dense_func
        deeplift.layers.convolutional.Conv2D._get_mxts_increments_for_inputs = saved_conv_func
        
        _pos_to_pos_mxts = True
        _neg_to_neg_mxts = True
        _neg_to_pos_mxts = True
        _pos_to_neg_mxts = True
        _zero_mxts = True
        
