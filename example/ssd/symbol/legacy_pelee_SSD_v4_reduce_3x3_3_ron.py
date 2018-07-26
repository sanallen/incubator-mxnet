import mxnet as mx
import numpy as np
import math
from common import legacy_conv_act_layer
from common import multibox_layer

# mxnet-cpu only support channel first, default convert the model and weight as channel first

def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=True,
					 nms_topk=400, **kwargs):
	
	data			= mx.sym.var('data')
	label = mx.symbol.Variable(name="label")	
	stem1		   = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2, 2), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, name = 'stem1')
	stem1_bn		= mx.sym.BatchNorm(data = stem1, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stem1/bn')
	stem1_relu	  = mx.sym.Activation(data = stem1_bn, act_type = 'relu', name = 'stem1/relu')
	stem2		   = mx.sym.Convolution(data=stem1_relu, kernel=(3, 3), stride=(2, 2), dilate = (), pad=(1, 1), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stem2')
	stem2_bn		= mx.sym.BatchNorm(data = stem2, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stem2/bn')
	stem2_relu	  = mx.sym.Activation(data = stem2_bn, act_type = 'relu', name = 'stem2/relu')
	stem3		   = mx.sym.Convolution(data=stem2_relu, kernel=(3, 3), stride=(2, 2), dilate = (), pad=(1, 1), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stem3')
	stem3_bn		= mx.sym.BatchNorm(data = stem3, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stem3/bn')
	stem3_relu	  = mx.sym.Activation(data = stem3_bn, act_type = 'relu', name = 'stem3/relu')
	stage2_1_branch2a = mx.sym.Convolution(data=stem3_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_1/branch2a')
	stage2_1_branch1a = mx.sym.Convolution(data=stem3_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_1/branch1a')
	stage2_1_branch2a_bn = mx.sym.BatchNorm(data = stage2_1_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_1/branch2a/bn')
	stage2_1_branch1a_bn = mx.sym.BatchNorm(data = stage2_1_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_1/branch1a/bn')
	stage2_1_branch2a_relu = mx.sym.Activation(data = stage2_1_branch2a_bn, act_type = 'relu', name = 'stage2_1/branch2a/relu')
	stage2_1_branch1a_relu = mx.sym.Activation(data = stage2_1_branch1a_bn, act_type = 'relu', name = 'stage2_1/branch1a/relu')
	stage2_1_branch2b = mx.sym.Convolution(data=stage2_1_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_1/branch2b')
	stage2_1_branch1b = mx.sym.Convolution(data=stage2_1_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_1/branch1b')
	stage2_1_branch2b_bn = mx.sym.BatchNorm(data = stage2_1_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_1/branch2b/bn')
	stage2_1_branch1b_bn = mx.sym.BatchNorm(data = stage2_1_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_1/branch1b/bn')
	stage2_1_branch2b_relu = mx.sym.Activation(data = stage2_1_branch2b_bn, act_type = 'relu', name = 'stage2_1/branch2b/relu')
	stage2_1_branch1b_relu = mx.sym.Activation(data = stage2_1_branch1b_bn, act_type = 'relu', name = 'stage2_1/branch1b/relu')
	stage2_1_branch2c = mx.sym.Convolution(data=stage2_1_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_1/branch2c')
	stage2_1_branch2c_bn = mx.sym.BatchNorm(data = stage2_1_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_1/branch2c/bn')
	stage2_1_branch2c_relu = mx.sym.Activation(data = stage2_1_branch2c_bn, act_type = 'relu', name = 'stage2_1/branch2c/relu')
	stage2_1_concat = mx.sym.concat(stem3_relu, stage2_1_branch1b_relu, stage2_1_branch2c_relu, dim = 1, name = 'stage2_1/concat')
	stage2_2_branch1a = mx.sym.Convolution(data=stage2_1_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_2/branch1a')
	stage2_2_branch2a = mx.sym.Convolution(data=stage2_1_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_2/branch2a')
	stage2_2_branch1a_bn = mx.sym.BatchNorm(data = stage2_2_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_2/branch1a/bn')
	stage2_2_branch2a_bn = mx.sym.BatchNorm(data = stage2_2_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_2/branch2a/bn')
	stage2_2_branch1a_relu = mx.sym.Activation(data = stage2_2_branch1a_bn, act_type = 'relu', name = 'stage2_2/branch1a/relu')
	stage2_2_branch2a_relu = mx.sym.Activation(data = stage2_2_branch2a_bn, act_type = 'relu', name = 'stage2_2/branch2a/relu')
	stage2_2_branch1b = mx.sym.Convolution(data=stage2_2_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_2/branch1b')
	stage2_2_branch2b = mx.sym.Convolution(data=stage2_2_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_2/branch2b')
	stage2_2_branch1b_bn = mx.sym.BatchNorm(data = stage2_2_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_2/branch1b/bn')
	stage2_2_branch2b_bn = mx.sym.BatchNorm(data = stage2_2_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_2/branch2b/bn')
	stage2_2_branch1b_relu = mx.sym.Activation(data = stage2_2_branch1b_bn, act_type = 'relu', name = 'stage2_2/branch1b/relu')
	stage2_2_branch2b_relu = mx.sym.Activation(data = stage2_2_branch2b_bn, act_type = 'relu', name = 'stage2_2/branch2b/relu')
	stage2_2_branch2c = mx.sym.Convolution(data=stage2_2_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_2/branch2c')
	stage2_2_branch2c_bn = mx.sym.BatchNorm(data = stage2_2_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_2/branch2c/bn')
	stage2_2_branch2c_relu = mx.sym.Activation(data = stage2_2_branch2c_bn, act_type = 'relu', name = 'stage2_2/branch2c/relu')
	stage2_2_concat = mx.sym.concat(stage2_1_concat, stage2_2_branch1b_relu, stage2_2_branch2c_relu, dim = 1, name = 'stage2_2/concat')
	stage2_3_branch1a = mx.sym.Convolution(data=stage2_2_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_3/branch1a')
	stage2_3_branch2a = mx.sym.Convolution(data=stage2_2_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_3/branch2a')
	stage2_3_branch1a_bn = mx.sym.BatchNorm(data = stage2_3_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_3/branch1a/bn')
	stage2_3_branch2a_bn = mx.sym.BatchNorm(data = stage2_3_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_3/branch2a/bn')
	stage2_3_branch1a_relu = mx.sym.Activation(data = stage2_3_branch1a_bn, act_type = 'relu', name = 'stage2_3/branch1a/relu')
	stage2_3_branch2a_relu = mx.sym.Activation(data = stage2_3_branch2a_bn, act_type = 'relu', name = 'stage2_3/branch2a/relu')
	stage2_3_branch1b = mx.sym.Convolution(data=stage2_3_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_3/branch1b')
	stage2_3_branch2b = mx.sym.Convolution(data=stage2_3_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_3/branch2b')
	stage2_3_branch1b_bn = mx.sym.BatchNorm(data = stage2_3_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_3/branch1b/bn')
	stage2_3_branch2b_bn = mx.sym.BatchNorm(data = stage2_3_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_3/branch2b/bn')
	stage2_3_branch1b_relu = mx.sym.Activation(data = stage2_3_branch1b_bn, act_type = 'relu', name = 'stage2_3/branch1b/relu')
	stage2_3_branch2b_relu = mx.sym.Activation(data = stage2_3_branch2b_bn, act_type = 'relu', name = 'stage2_3/branch2b/relu')
	stage2_3_branch2c = mx.sym.Convolution(data=stage2_3_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_3/branch2c')
	stage2_3_branch2c_bn = mx.sym.BatchNorm(data = stage2_3_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_3/branch2c/bn')
	stage2_3_branch2c_relu = mx.sym.Activation(data = stage2_3_branch2c_bn, act_type = 'relu', name = 'stage2_3/branch2c/relu')
	stage2_3_concat = mx.sym.concat(stage2_2_concat, stage2_3_branch1b_relu, stage2_3_branch2c_relu, dim = 1, name = 'stage2_3/concat')

	stage2_4_branch1a = mx.sym.Convolution(data=stage2_3_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_4/branch1a')
	stage2_4_branch2a = mx.sym.Convolution(data=stage2_3_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_4/branch2a')
	stage2_4_branch1a_bn = mx.sym.BatchNorm(data = stage2_4_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_4/branch1a/bn')
	stage2_4_branch2a_bn = mx.sym.BatchNorm(data = stage2_4_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_4/branch2a/bn')
	stage2_4_branch1a_relu = mx.sym.Activation(data = stage2_4_branch1a_bn, act_type = 'relu', name = 'stage2_4/branch1a/relu')
	stage2_4_branch2a_relu = mx.sym.Activation(data = stage2_4_branch2a_bn, act_type = 'relu', name = 'stage2_4/branch2a/relu')
	stage2_4_branch1b = mx.sym.Convolution(data=stage2_4_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_4/branch1b')
	stage2_4_branch2b = mx.sym.Convolution(data=stage2_4_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_4/branch2b')
	stage2_4_branch1b_bn = mx.sym.BatchNorm(data = stage2_4_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_4/branch1b/bn')
	stage2_4_branch2b_bn = mx.sym.BatchNorm(data = stage2_4_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_4/branch2b/bn')
	stage2_4_branch1b_relu = mx.sym.Activation(data = stage2_4_branch1b_bn, act_type = 'relu', name = 'stage2_4/branch1b/relu')
	stage2_4_branch2b_relu = mx.sym.Activation(data = stage2_4_branch2b_bn, act_type = 'relu', name = 'stage2_4/branch2b/relu')
	stage2_4_branch2c = mx.sym.Convolution(data=stage2_4_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_4/branch2c')
	stage2_4_branch2c_bn = mx.sym.BatchNorm(data = stage2_4_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_4/branch2c/bn')
	stage2_4_branch2c_relu = mx.sym.Activation(data = stage2_4_branch2c_bn, act_type = 'relu', name = 'stage2_4/branch2c/relu')
	stage2_4_concat = mx.sym.concat(stage2_3_concat, stage2_4_branch1b_relu, stage2_4_branch2c_relu, dim = 1, name = 'stage2_4/concat')

	stage2_5_branch1a = mx.sym.Convolution(data=stage2_4_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_5/branch1a')
	stage2_5_branch2a = mx.sym.Convolution(data=stage2_4_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_5/branch2a')
	stage2_5_branch1a_bn = mx.sym.BatchNorm(data = stage2_5_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_5/branch1a/bn')
	stage2_5_branch2a_bn = mx.sym.BatchNorm(data = stage2_5_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_5/branch2a/bn')
	stage2_5_branch1a_relu = mx.sym.Activation(data = stage2_5_branch1a_bn, act_type = 'relu', name = 'stage2_5/branch1a/relu')
	stage2_5_branch2a_relu = mx.sym.Activation(data = stage2_5_branch2a_bn, act_type = 'relu', name = 'stage2_5/branch2a/relu')
	stage2_5_branch1b = mx.sym.Convolution(data=stage2_5_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_5/branch1b')
	stage2_5_branch2b = mx.sym.Convolution(data=stage2_5_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_5/branch2b')
	stage2_5_branch1b_bn = mx.sym.BatchNorm(data = stage2_5_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_5/branch1b/bn')
	stage2_5_branch2b_bn = mx.sym.BatchNorm(data = stage2_5_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_5/branch2b/bn')
	stage2_5_branch1b_relu = mx.sym.Activation(data = stage2_5_branch1b_bn, act_type = 'relu', name = 'stage2_5/branch1b/relu')
	stage2_5_branch2b_relu = mx.sym.Activation(data = stage2_5_branch2b_bn, act_type = 'relu', name = 'stage2_5/branch2b/relu')
	stage2_5_branch2c = mx.sym.Convolution(data=stage2_5_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_5/branch2c')
	stage2_5_branch2c_bn = mx.sym.BatchNorm(data = stage2_5_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_5/branch2c/bn')
	stage2_5_branch2c_relu = mx.sym.Activation(data = stage2_5_branch2c_bn, act_type = 'relu', name = 'stage2_5/branch2c/relu')
	stage2_5_concat = mx.sym.concat(stage2_4_concat, stage2_5_branch1b_relu, stage2_5_branch2c_relu, dim = 1, name = 'stage2_5/concat')

	stage2_6_branch1a = mx.sym.Convolution(data=stage2_5_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_6/branch1a')
	stage2_6_branch2a = mx.sym.Convolution(data=stage2_5_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_6/branch2a')
	stage2_6_branch1a_bn = mx.sym.BatchNorm(data = stage2_6_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_6/branch1a/bn')
	stage2_6_branch2a_bn = mx.sym.BatchNorm(data = stage2_6_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_6/branch2a/bn')
	stage2_6_branch1a_relu = mx.sym.Activation(data = stage2_6_branch1a_bn, act_type = 'relu', name = 'stage2_6/branch1a/relu')
	stage2_6_branch2a_relu = mx.sym.Activation(data = stage2_6_branch2a_bn, act_type = 'relu', name = 'stage2_6/branch2a/relu')
	stage2_6_branch1b = mx.sym.Convolution(data=stage2_6_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_6/branch1b')
	stage2_6_branch2b = mx.sym.Convolution(data=stage2_6_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_6/branch2b')
	stage2_6_branch1b_bn = mx.sym.BatchNorm(data = stage2_6_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_6/branch1b/bn')
	stage2_6_branch2b_bn = mx.sym.BatchNorm(data = stage2_6_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_6/branch2b/bn')
	stage2_6_branch1b_relu = mx.sym.Activation(data = stage2_6_branch1b_bn, act_type = 'relu', name = 'stage2_6/branch1b/relu')
	stage2_6_branch2b_relu = mx.sym.Activation(data = stage2_6_branch2b_bn, act_type = 'relu', name = 'stage2_6/branch2b/relu')
	stage2_6_branch2c = mx.sym.Convolution(data=stage2_6_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_6/branch2c')
	stage2_6_branch2c_bn = mx.sym.BatchNorm(data = stage2_6_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_6/branch2c/bn')
	stage2_6_branch2c_relu = mx.sym.Activation(data = stage2_6_branch2c_bn, act_type = 'relu', name = 'stage2_6/branch2c/relu')
	stage2_6_concat = mx.sym.concat(stage2_5_concat, stage2_6_branch1b_relu, stage2_6_branch2c_relu, dim = 1, name = 'stage2_6/concat')
	
	stage2_tb	   = mx.sym.Convolution(data=stage2_6_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage2_tb')
	stage2_tb_bn	= mx.sym.BatchNorm(data = stage2_tb, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage2_tb/bn')
	stage2_tb_relu  = mx.sym.Activation(data = stage2_tb_bn, act_type = 'relu', name = 'stage2_tb/relu')
	stage2_tb_pool  = mx.sym.Pooling(data = stage2_tb_relu, global_pool = False, kernel=(2, 2), pool_type = 'avg', stride=(2, 2), name = 'stage2_tb/pool')
	stage3_1_branch1a = mx.sym.Convolution(data=stage2_tb_pool, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_1/branch1a')
	stage3_1_branch2a = mx.sym.Convolution(data=stage2_tb_pool, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_1/branch2a')
	stage3_1_branch1a_bn = mx.sym.BatchNorm(data = stage3_1_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_1/branch1a/bn')
	stage3_1_branch2a_bn = mx.sym.BatchNorm(data = stage3_1_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_1/branch2a/bn')
	stage3_1_branch1a_relu = mx.sym.Activation(data = stage3_1_branch1a_bn, act_type = 'relu', name = 'stage3_1/branch1a/relu')
	stage3_1_branch2a_relu = mx.sym.Activation(data = stage3_1_branch2a_bn, act_type = 'relu', name = 'stage3_1/branch2a/relu')
	stage3_1_branch1b = mx.sym.Convolution(data=stage3_1_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_1/branch1b')
	stage3_1_branch2b = mx.sym.Convolution(data=stage3_1_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_1/branch2b')
	stage3_1_branch1b_bn = mx.sym.BatchNorm(data = stage3_1_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_1/branch1b/bn')
	stage3_1_branch2b_bn = mx.sym.BatchNorm(data = stage3_1_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_1/branch2b/bn')
	stage3_1_branch1b_relu = mx.sym.Activation(data = stage3_1_branch1b_bn, act_type = 'relu', name = 'stage3_1/branch1b/relu')
	stage3_1_branch2b_relu = mx.sym.Activation(data = stage3_1_branch2b_bn, act_type = 'relu', name = 'stage3_1/branch2b/relu')
	stage3_1_branch2c = mx.sym.Convolution(data=stage3_1_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_1/branch2c')
	stage3_1_branch2c_bn = mx.sym.BatchNorm(data = stage3_1_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_1/branch2c/bn')
	stage3_1_branch2c_relu = mx.sym.Activation(data = stage3_1_branch2c_bn, act_type = 'relu', name = 'stage3_1/branch2c/relu')
	stage3_1_concat = mx.sym.concat(stage2_tb_pool, stage3_1_branch1b_relu, stage3_1_branch2c_relu, dim = 1, name = 'stage3_1/concat')
	stage3_2_branch2a = mx.sym.Convolution(data=stage3_1_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_2/branch2a')
	stage3_2_branch1a = mx.sym.Convolution(data=stage3_1_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_2/branch1a')
	stage3_2_branch2a_bn = mx.sym.BatchNorm(data = stage3_2_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_2/branch2a/bn')
	stage3_2_branch1a_bn = mx.sym.BatchNorm(data = stage3_2_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_2/branch1a/bn')
	stage3_2_branch2a_relu = mx.sym.Activation(data = stage3_2_branch2a_bn, act_type = 'relu', name = 'stage3_2/branch2a/relu')
	stage3_2_branch1a_relu = mx.sym.Activation(data = stage3_2_branch1a_bn, act_type = 'relu', name = 'stage3_2/branch1a/relu')
	stage3_2_branch2b = mx.sym.Convolution(data=stage3_2_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_2/branch2b')
	stage3_2_branch1b = mx.sym.Convolution(data=stage3_2_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_2/branch1b')
	stage3_2_branch2b_bn = mx.sym.BatchNorm(data = stage3_2_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_2/branch2b/bn')
	stage3_2_branch1b_bn = mx.sym.BatchNorm(data = stage3_2_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_2/branch1b/bn')
	stage3_2_branch2b_relu = mx.sym.Activation(data = stage3_2_branch2b_bn, act_type = 'relu', name = 'stage3_2/branch2b/relu')
	stage3_2_branch1b_relu = mx.sym.Activation(data = stage3_2_branch1b_bn, act_type = 'relu', name = 'stage3_2/branch1b/relu')
	stage3_2_branch2c = mx.sym.Convolution(data=stage3_2_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_2/branch2c')
	stage3_2_branch2c_bn = mx.sym.BatchNorm(data = stage3_2_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_2/branch2c/bn')
	stage3_2_branch2c_relu = mx.sym.Activation(data = stage3_2_branch2c_bn, act_type = 'relu', name = 'stage3_2/branch2c/relu')
	stage3_2_concat = mx.sym.concat(stage3_1_concat, stage3_2_branch1b_relu, stage3_2_branch2c_relu, dim = 1, name = 'stage3_2/concat')
	stage3_3_branch2a = mx.sym.Convolution(data=stage3_2_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_3/branch2a')
	stage3_3_branch1a = mx.sym.Convolution(data=stage3_2_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_3/branch1a')
	stage3_3_branch2a_bn = mx.sym.BatchNorm(data = stage3_3_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_3/branch2a/bn')
	stage3_3_branch1a_bn = mx.sym.BatchNorm(data = stage3_3_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_3/branch1a/bn')
	stage3_3_branch2a_relu = mx.sym.Activation(data = stage3_3_branch2a_bn, act_type = 'relu', name = 'stage3_3/branch2a/relu')
	stage3_3_branch1a_relu = mx.sym.Activation(data = stage3_3_branch1a_bn, act_type = 'relu', name = 'stage3_3/branch1a/relu')
	stage3_3_branch2b = mx.sym.Convolution(data=stage3_3_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_3/branch2b')
	stage3_3_branch1b = mx.sym.Convolution(data=stage3_3_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_3/branch1b')
	stage3_3_branch2b_bn = mx.sym.BatchNorm(data = stage3_3_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_3/branch2b/bn')
	stage3_3_branch1b_bn = mx.sym.BatchNorm(data = stage3_3_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_3/branch1b/bn')
	stage3_3_branch2b_relu = mx.sym.Activation(data = stage3_3_branch2b_bn, act_type = 'relu', name = 'stage3_3/branch2b/relu')
	stage3_3_branch1b_relu = mx.sym.Activation(data = stage3_3_branch1b_bn, act_type = 'relu', name = 'stage3_3/branch1b/relu')
	stage3_3_branch2c = mx.sym.Convolution(data=stage3_3_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_3/branch2c')
	stage3_3_branch2c_bn = mx.sym.BatchNorm(data = stage3_3_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_3/branch2c/bn')
	stage3_3_branch2c_relu = mx.sym.Activation(data = stage3_3_branch2c_bn, act_type = 'relu', name = 'stage3_3/branch2c/relu')
	stage3_3_concat = mx.sym.concat(stage3_2_concat, stage3_3_branch1b_relu, stage3_3_branch2c_relu, dim = 1, name = 'stage3_3/concat')
	stage3_4_branch1a = mx.sym.Convolution(data=stage3_3_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_4/branch1a')
	stage3_4_branch2a = mx.sym.Convolution(data=stage3_3_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_4/branch2a')
	stage3_4_branch1a_bn = mx.sym.BatchNorm(data = stage3_4_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_4/branch1a/bn')
	stage3_4_branch2a_bn = mx.sym.BatchNorm(data = stage3_4_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_4/branch2a/bn')
	stage3_4_branch1a_relu = mx.sym.Activation(data = stage3_4_branch1a_bn, act_type = 'relu', name = 'stage3_4/branch1a/relu')
	stage3_4_branch2a_relu = mx.sym.Activation(data = stage3_4_branch2a_bn, act_type = 'relu', name = 'stage3_4/branch2a/relu')
	stage3_4_branch1b = mx.sym.Convolution(data=stage3_4_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_4/branch1b')
	stage3_4_branch2b = mx.sym.Convolution(data=stage3_4_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_4/branch2b')
	stage3_4_branch1b_bn = mx.sym.BatchNorm(data = stage3_4_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_4/branch1b/bn')
	stage3_4_branch2b_bn = mx.sym.BatchNorm(data = stage3_4_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_4/branch2b/bn')
	stage3_4_branch1b_relu = mx.sym.Activation(data = stage3_4_branch1b_bn, act_type = 'relu', name = 'stage3_4/branch1b/relu')
	stage3_4_branch2b_relu = mx.sym.Activation(data = stage3_4_branch2b_bn, act_type = 'relu', name = 'stage3_4/branch2b/relu')
	stage3_4_branch2c = mx.sym.Convolution(data=stage3_4_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_4/branch2c')
	stage3_4_branch2c_bn = mx.sym.BatchNorm(data = stage3_4_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_4/branch2c/bn')
	stage3_4_branch2c_relu = mx.sym.Activation(data = stage3_4_branch2c_bn, act_type = 'relu', name = 'stage3_4/branch2c/relu')
	stage3_4_concat = mx.sym.concat(stage3_3_concat, stage3_4_branch1b_relu, stage3_4_branch2c_relu, dim = 1, name = 'stage3_4/concat')
	
	stage3_5_branch1a = mx.sym.Convolution(data=stage3_4_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_5/branch1a')
	stage3_5_branch2a = mx.sym.Convolution(data=stage3_4_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_5/branch2a')
	stage3_5_branch1a_bn = mx.sym.BatchNorm(data = stage3_5_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_5/branch1a/bn')
	stage3_5_branch2a_bn = mx.sym.BatchNorm(data = stage3_5_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_5/branch2a/bn')
	stage3_5_branch1a_relu = mx.sym.Activation(data = stage3_5_branch1a_bn, act_type = 'relu', name = 'stage3_5/branch1a/relu')
	stage3_5_branch2a_relu = mx.sym.Activation(data = stage3_5_branch2a_bn, act_type = 'relu', name = 'stage3_5/branch2a/relu')
	stage3_5_branch1b = mx.sym.Convolution(data=stage3_5_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_5/branch1b')
	stage3_5_branch2b = mx.sym.Convolution(data=stage3_5_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_5/branch2b')
	stage3_5_branch1b_bn = mx.sym.BatchNorm(data = stage3_5_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_5/branch1b/bn')
	stage3_5_branch2b_bn = mx.sym.BatchNorm(data = stage3_5_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_5/branch2b/bn')
	stage3_5_branch1b_relu = mx.sym.Activation(data = stage3_5_branch1b_bn, act_type = 'relu', name = 'stage3_5/branch1b/relu')
	stage3_5_branch2b_relu = mx.sym.Activation(data = stage3_5_branch2b_bn, act_type = 'relu', name = 'stage3_5/branch2b/relu')
	stage3_5_branch2c = mx.sym.Convolution(data=stage3_5_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_5/branch2c')
	stage3_5_branch2c_bn = mx.sym.BatchNorm(data = stage3_5_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_5/branch2c/bn')
	stage3_5_branch2c_relu = mx.sym.Activation(data = stage3_5_branch2c_bn, act_type = 'relu', name = 'stage3_5/branch2c/relu')
	stage3_5_concat = mx.sym.concat(stage3_4_concat, stage3_5_branch1b_relu, stage3_5_branch2c_relu, dim = 1, name = 'stage3_5/concat')

	stage3_6_branch1a = mx.sym.Convolution(data=stage3_5_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_6/branch1a')
	stage3_6_branch2a = mx.sym.Convolution(data=stage3_5_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_6/branch2a')
	stage3_6_branch1a_bn = mx.sym.BatchNorm(data = stage3_6_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_6/branch1a/bn')
	stage3_6_branch2a_bn = mx.sym.BatchNorm(data = stage3_6_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_6/branch2a/bn')
	stage3_6_branch1a_relu = mx.sym.Activation(data = stage3_6_branch1a_bn, act_type = 'relu', name = 'stage3_6/branch1a/relu')
	stage3_6_branch2a_relu = mx.sym.Activation(data = stage3_6_branch2a_bn, act_type = 'relu', name = 'stage3_6/branch2a/relu')
	stage3_6_branch1b = mx.sym.Convolution(data=stage3_6_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_6/branch1b')
	stage3_6_branch2b = mx.sym.Convolution(data=stage3_6_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_6/branch2b')
	stage3_6_branch1b_bn = mx.sym.BatchNorm(data = stage3_6_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_6/branch1b/bn')
	stage3_6_branch2b_bn = mx.sym.BatchNorm(data = stage3_6_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_6/branch2b/bn')
	stage3_6_branch1b_relu = mx.sym.Activation(data = stage3_6_branch1b_bn, act_type = 'relu', name = 'stage3_6/branch1b/relu')
	stage3_6_branch2b_relu = mx.sym.Activation(data = stage3_6_branch2b_bn, act_type = 'relu', name = 'stage3_6/branch2b/relu')
	stage3_6_branch2c = mx.sym.Convolution(data=stage3_6_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_6/branch2c')
	stage3_6_branch2c_bn = mx.sym.BatchNorm(data = stage3_6_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_6/branch2c/bn')
	stage3_6_branch2c_relu = mx.sym.Activation(data = stage3_6_branch2c_bn, act_type = 'relu', name = 'stage3_6/branch2c/relu')
	stage3_6_concat = mx.sym.concat(stage3_5_concat, stage3_6_branch1b_relu, stage3_6_branch2c_relu, dim = 1, name = 'stage3_6/concat')

	
	stage3_tb	   = mx.sym.Convolution(data=stage3_6_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage3_tb')
	stage3_tb_bn	= mx.sym.BatchNorm(data = stage3_tb, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage3_tb/bn')
	stage3_tb_relu  = mx.sym.Activation(data = stage3_tb_bn, act_type = 'relu', name = 'stage3_tb/relu')
	stage3_tb_pool  = mx.sym.Pooling(data = stage3_tb_relu, global_pool = False, kernel=(2, 2), pool_type = 'avg', stride=(2, 2), name = 'stage3_tb/pool')
	stage4_tb_ext_pm2_b2a = mx.sym.Convolution(data=stage3_tb_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm2/b2a')
	stage4_tb_ext_pm2 = mx.sym.Convolution(data=stage3_tb_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm2')
	stage4_1_branch2a = mx.sym.Convolution(data=stage3_tb_pool, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_1/branch2a')
	stage4_1_branch1a = mx.sym.Convolution(data=stage3_tb_pool, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_1/branch1a')
	stage4_tb_ext_pm2_b2a_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm2_b2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm2/b2a/bn')
	stage4_tb_ext_pm2_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm2, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm2/bn')
	stage4_1_branch2a_bn = mx.sym.BatchNorm(data = stage4_1_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_1/branch2a/bn')
	stage4_1_branch1a_bn = mx.sym.BatchNorm(data = stage4_1_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_1/branch1a/bn')
	stage4_tb_ext_pm2_b2a_relu = mx.sym.Activation(data = stage4_tb_ext_pm2_b2a_bn, act_type = 'relu', name = 'stage4_tb/ext/pm2/b2a/relu')
	stage4_1_branch2a_relu = mx.sym.Activation(data = stage4_1_branch2a_bn, act_type = 'relu', name = 'stage4_1/branch2a/relu')
	stage4_1_branch1a_relu = mx.sym.Activation(data = stage4_1_branch1a_bn, act_type = 'relu', name = 'stage4_1/branch1a/relu')
	stage4_tb_ext_pm2_b2b = mx.sym.Convolution(data=stage4_tb_ext_pm2_b2a_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm2/b2b')
	stage4_1_branch2b = mx.sym.Convolution(data=stage4_1_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_1/branch2b')
	stage4_1_branch1b = mx.sym.Convolution(data=stage4_1_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_1/branch1b')
	stage4_tb_ext_pm2_b2b_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm2_b2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm2/b2b/bn')
	stage4_1_branch2b_bn = mx.sym.BatchNorm(data = stage4_1_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_1/branch2b/bn')
	stage4_1_branch1b_bn = mx.sym.BatchNorm(data = stage4_1_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_1/branch1b/bn')
	stage4_tb_ext_pm2_b2b_relu = mx.sym.Activation(data = stage4_tb_ext_pm2_b2b_bn, act_type = 'relu', name = 'stage4_tb/ext/pm2/b2b/relu')
	stage4_1_branch2b_relu = mx.sym.Activation(data = stage4_1_branch2b_bn, act_type = 'relu', name = 'stage4_1/branch2b/relu')
	stage4_1_branch1b_relu = mx.sym.Activation(data = stage4_1_branch1b_bn, act_type = 'relu', name = 'stage4_1/branch1b/relu')
	stage4_tb_ext_pm2_b2c = mx.sym.Convolution(data=stage4_tb_ext_pm2_b2b_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm2/b2c')
	stage4_1_branch2c = mx.sym.Convolution(data=stage4_1_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_1/branch2c')
	stage4_tb_ext_pm2_b2c_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm2_b2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm2/b2c/bn')
	stage4_1_branch2c_bn = mx.sym.BatchNorm(data = stage4_1_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_1/branch2c/bn')
	stage4_tb_ext_pm2_res = mx.sym.broadcast_add(stage4_tb_ext_pm2_bn, stage4_tb_ext_pm2_b2c_bn)
	stage4_1_branch2c_relu = mx.sym.Activation(data = stage4_1_branch2c_bn, act_type = 'relu', name = 'stage4_1/branch2c/relu')
	stage4_tb_ext_pm2_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm2_res, act_type = 'relu', name = 'stage4_tb/ext/pm2/res/relu')
	stage4_1_concat = mx.sym.concat(stage3_tb_pool, stage4_1_branch1b_relu, stage4_1_branch2c_relu, dim = 1, name = 'stage4_1/concat')
	stage4_2_branch1a = mx.sym.Convolution(data=stage4_1_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_2/branch1a')
	stage4_2_branch2a = mx.sym.Convolution(data=stage4_1_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_2/branch2a')
	stage4_2_branch1a_bn = mx.sym.BatchNorm(data = stage4_2_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_2/branch1a/bn')
	stage4_2_branch2a_bn = mx.sym.BatchNorm(data = stage4_2_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_2/branch2a/bn')
	stage4_2_branch1a_relu = mx.sym.Activation(data = stage4_2_branch1a_bn, act_type = 'relu', name = 'stage4_2/branch1a/relu')
	stage4_2_branch2a_relu = mx.sym.Activation(data = stage4_2_branch2a_bn, act_type = 'relu', name = 'stage4_2/branch2a/relu')
	stage4_2_branch1b = mx.sym.Convolution(data=stage4_2_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_2/branch1b')
	stage4_2_branch2b = mx.sym.Convolution(data=stage4_2_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_2/branch2b')
	stage4_2_branch1b_bn = mx.sym.BatchNorm(data = stage4_2_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_2/branch1b/bn')
	stage4_2_branch2b_bn = mx.sym.BatchNorm(data = stage4_2_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_2/branch2b/bn')
	stage4_2_branch1b_relu = mx.sym.Activation(data = stage4_2_branch1b_bn, act_type = 'relu', name = 'stage4_2/branch1b/relu')
	stage4_2_branch2b_relu = mx.sym.Activation(data = stage4_2_branch2b_bn, act_type = 'relu', name = 'stage4_2/branch2b/relu')
	stage4_2_branch2c = mx.sym.Convolution(data=stage4_2_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_2/branch2c')
	stage4_2_branch2c_bn = mx.sym.BatchNorm(data = stage4_2_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_2/branch2c/bn')
	stage4_2_branch2c_relu = mx.sym.Activation(data = stage4_2_branch2c_bn, act_type = 'relu', name = 'stage4_2/branch2c/relu')
	stage4_2_concat = mx.sym.concat(stage4_1_concat, stage4_2_branch1b_relu, stage4_2_branch2c_relu, dim = 1, name = 'stage4_2/concat')
	stage4_3_branch2a = mx.sym.Convolution(data=stage4_2_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_3/branch2a')
	stage4_3_branch1a = mx.sym.Convolution(data=stage4_2_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_3/branch1a')
	stage4_3_branch2a_bn = mx.sym.BatchNorm(data = stage4_3_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_3/branch2a/bn')
	stage4_3_branch1a_bn = mx.sym.BatchNorm(data = stage4_3_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_3/branch1a/bn')
	stage4_3_branch2a_relu = mx.sym.Activation(data = stage4_3_branch2a_bn, act_type = 'relu', name = 'stage4_3/branch2a/relu')
	stage4_3_branch1a_relu = mx.sym.Activation(data = stage4_3_branch1a_bn, act_type = 'relu', name = 'stage4_3/branch1a/relu')
	stage4_3_branch2b = mx.sym.Convolution(data=stage4_3_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_3/branch2b')
	stage4_3_branch1b = mx.sym.Convolution(data=stage4_3_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_3/branch1b')
	stage4_3_branch2b_bn = mx.sym.BatchNorm(data = stage4_3_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_3/branch2b/bn')
	stage4_3_branch1b_bn = mx.sym.BatchNorm(data = stage4_3_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_3/branch1b/bn')
	stage4_3_branch2b_relu = mx.sym.Activation(data = stage4_3_branch2b_bn, act_type = 'relu', name = 'stage4_3/branch2b/relu')
	stage4_3_branch1b_relu = mx.sym.Activation(data = stage4_3_branch1b_bn, act_type = 'relu', name = 'stage4_3/branch1b/relu')
	stage4_3_branch2c = mx.sym.Convolution(data=stage4_3_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_3/branch2c')
	stage4_3_branch2c_bn = mx.sym.BatchNorm(data = stage4_3_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_3/branch2c/bn')
	stage4_3_branch2c_relu = mx.sym.Activation(data = stage4_3_branch2c_bn, act_type = 'relu', name = 'stage4_3/branch2c/relu')
	stage4_3_concat = mx.sym.concat(stage4_2_concat, stage4_3_branch1b_relu, stage4_3_branch2c_relu, dim = 1, name = 'stage4_3/concat')
	stage4_4_branch1a = mx.sym.Convolution(data=stage4_3_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_4/branch1a')
	stage4_4_branch2a = mx.sym.Convolution(data=stage4_3_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_4/branch2a')
	stage4_4_branch1a_bn = mx.sym.BatchNorm(data = stage4_4_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_4/branch1a/bn')
	stage4_4_branch2a_bn = mx.sym.BatchNorm(data = stage4_4_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_4/branch2a/bn')
	stage4_4_branch1a_relu = mx.sym.Activation(data = stage4_4_branch1a_bn, act_type = 'relu', name = 'stage4_4/branch1a/relu')
	stage4_4_branch2a_relu = mx.sym.Activation(data = stage4_4_branch2a_bn, act_type = 'relu', name = 'stage4_4/branch2a/relu')
	stage4_4_branch1b = mx.sym.Convolution(data=stage4_4_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_4/branch1b')
	stage4_4_branch2b = mx.sym.Convolution(data=stage4_4_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_4/branch2b')
	stage4_4_branch1b_bn = mx.sym.BatchNorm(data = stage4_4_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_4/branch1b/bn')
	stage4_4_branch2b_bn = mx.sym.BatchNorm(data = stage4_4_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_4/branch2b/bn')
	stage4_4_branch1b_relu = mx.sym.Activation(data = stage4_4_branch1b_bn, act_type = 'relu', name = 'stage4_4/branch1b/relu')
	stage4_4_branch2b_relu = mx.sym.Activation(data = stage4_4_branch2b_bn, act_type = 'relu', name = 'stage4_4/branch2b/relu')
	stage4_4_branch2c = mx.sym.Convolution(data=stage4_4_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_4/branch2c')
	stage4_4_branch2c_bn = mx.sym.BatchNorm(data = stage4_4_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_4/branch2c/bn')
	stage4_4_branch2c_relu = mx.sym.Activation(data = stage4_4_branch2c_bn, act_type = 'relu', name = 'stage4_4/branch2c/relu')
	stage4_4_concat = mx.sym.concat(stage4_3_concat, stage4_4_branch1b_relu, stage4_4_branch2c_relu, dim = 1, name = 'stage4_4/concat')
	stage4_5_branch2a = mx.sym.Convolution(data=stage4_4_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_5/branch2a')
	stage4_5_branch1a = mx.sym.Convolution(data=stage4_4_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_5/branch1a')
	stage4_5_branch2a_bn = mx.sym.BatchNorm(data = stage4_5_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_5/branch2a/bn')
	stage4_5_branch1a_bn = mx.sym.BatchNorm(data = stage4_5_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_5/branch1a/bn')
	stage4_5_branch2a_relu = mx.sym.Activation(data = stage4_5_branch2a_bn, act_type = 'relu', name = 'stage4_5/branch2a/relu')
	stage4_5_branch1a_relu = mx.sym.Activation(data = stage4_5_branch1a_bn, act_type = 'relu', name = 'stage4_5/branch1a/relu')
	stage4_5_branch2b = mx.sym.Convolution(data=stage4_5_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_5/branch2b')
	stage4_5_branch1b = mx.sym.Convolution(data=stage4_5_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_5/branch1b')
	stage4_5_branch2b_bn = mx.sym.BatchNorm(data = stage4_5_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_5/branch2b/bn')
	stage4_5_branch1b_bn = mx.sym.BatchNorm(data = stage4_5_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_5/branch1b/bn')
	stage4_5_branch2b_relu = mx.sym.Activation(data = stage4_5_branch2b_bn, act_type = 'relu', name = 'stage4_5/branch2b/relu')
	stage4_5_branch1b_relu = mx.sym.Activation(data = stage4_5_branch1b_bn, act_type = 'relu', name = 'stage4_5/branch1b/relu')
	stage4_5_branch2c = mx.sym.Convolution(data=stage4_5_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_5/branch2c')
	stage4_5_branch2c_bn = mx.sym.BatchNorm(data = stage4_5_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_5/branch2c/bn')
	stage4_5_branch2c_relu = mx.sym.Activation(data = stage4_5_branch2c_bn, act_type = 'relu', name = 'stage4_5/branch2c/relu')
	stage4_5_concat = mx.sym.concat(stage4_4_concat, stage4_5_branch1b_relu, stage4_5_branch2c_relu, dim = 1, name = 'stage4_5/concat')
	stage4_6_branch1a = mx.sym.Convolution(data=stage4_5_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_6/branch1a')
	stage4_6_branch2a = mx.sym.Convolution(data=stage4_5_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 64, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_6/branch2a')
	stage4_6_branch1a_bn = mx.sym.BatchNorm(data = stage4_6_branch1a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_6/branch1a/bn')
	stage4_6_branch2a_bn = mx.sym.BatchNorm(data = stage4_6_branch2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_6/branch2a/bn')
	stage4_6_branch1a_relu = mx.sym.Activation(data = stage4_6_branch1a_bn, act_type = 'relu', name = 'stage4_6/branch1a/relu')
	stage4_6_branch2a_relu = mx.sym.Activation(data = stage4_6_branch2a_bn, act_type = 'relu', name = 'stage4_6/branch2a/relu')
	stage4_6_branch1b = mx.sym.Convolution(data=stage4_6_branch1a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_6/branch1b')
	stage4_6_branch2b = mx.sym.Convolution(data=stage4_6_branch2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_6/branch2b')
	stage4_6_branch1b_bn = mx.sym.BatchNorm(data = stage4_6_branch1b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_6/branch1b/bn')
	stage4_6_branch2b_bn = mx.sym.BatchNorm(data = stage4_6_branch2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_6/branch2b/bn')
	stage4_6_branch1b_relu = mx.sym.Activation(data = stage4_6_branch1b_bn, act_type = 'relu', name = 'stage4_6/branch1b/relu')
	stage4_6_branch2b_relu = mx.sym.Activation(data = stage4_6_branch2b_bn, act_type = 'relu', name = 'stage4_6/branch2b/relu')
	stage4_6_branch2c = mx.sym.Convolution(data=stage4_6_branch2b_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_6/branch2c')
	stage4_6_branch2c_bn = mx.sym.BatchNorm(data = stage4_6_branch2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_6/branch2c/bn')
	stage4_6_branch2c_relu = mx.sym.Activation(data = stage4_6_branch2c_bn, act_type = 'relu', name = 'stage4_6/branch2c/relu')
	stage4_6_concat = mx.sym.concat(stage4_5_concat, stage4_6_branch1b_relu, stage4_6_branch2c_relu, dim = 1, name = 'stage4_6/concat')
	stage4_tb	   = mx.sym.Convolution(data=stage4_6_concat, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 448, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb')
	stage4_tb_bn	= mx.sym.BatchNorm(data = stage4_tb, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/bn')
	stage4_tb_relu  = mx.sym.Activation(data = stage4_tb_bn, act_type = 'relu', name = 'stage4_tb/relu')
	stage4_tb_relu_ext1_fe1_1 = mx.sym.Convolution(data=stage4_tb_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'stage4_tb/relu/ext1/fe1_1')
	stage4_tb_ext_pm3_b2a = mx.sym.Convolution(data=stage4_tb_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm3/b2a')
	stage4_tb_ext_pm3 = mx.sym.Convolution(data=stage4_tb_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm3')
	stage4_tb_relu_ext1_fe1_1_relu = mx.sym.Activation(data = stage4_tb_relu_ext1_fe1_1, act_type = 'relu', name = 'stage4_tb/relu/ext1/fe1_1/relu')
	stage4_tb_ext_pm3_b2a_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3_b2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm3/b2a/bn')
	stage4_tb_ext_pm3_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm3/bn')
	ext1_fe1_2	  = mx.sym.Convolution(data=stage4_tb_relu_ext1_fe1_1_relu, kernel=(3, 3), stride=(2, 2), dilate = (), pad=(1, 1), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'ext1/fe1_2')
	stage4_tb_ext_pm3_b2a_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_b2a_bn, act_type = 'relu', name = 'stage4_tb/ext/pm3/b2a/relu')
	ext1_fe1_2_relu = mx.sym.Activation(data = ext1_fe1_2, act_type = 'relu', name = 'ext1/fe1_2/relu')
	stage4_tb_ext_pm3_b2b = mx.sym.Convolution(data=stage4_tb_ext_pm3_b2a_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm3/b2b')
	ext1_fe2_1	  = mx.sym.Convolution(data=ext1_fe1_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = False, layout = 'NCHW', name = 'ext1/fe2_1')
	stage4_tb_ext_pm4_b2a = mx.sym.Convolution(data=ext1_fe1_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm4/b2a')
	stage4_tb_ext_pm4 = mx.sym.Convolution(data=ext1_fe1_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm4')
	stage4_tb_ext_pm3_b2b_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3_b2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm3/b2b/bn')
	ext1_fe2_1_relu = mx.sym.Activation(data = ext1_fe2_1, act_type = 'relu', name = 'ext1/fe2_1/relu')
	stage4_tb_ext_pm4_b2a_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm4_b2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm4/b2a/bn')
	stage4_tb_ext_pm4_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm4, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm4/bn')
	stage4_tb_ext_pm3_b2b_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_b2b_bn, act_type = 'relu', name = 'stage4_tb/ext/pm3/b2b/relu')
	ext1_fe2_2	  = mx.sym.Convolution(data=ext1_fe2_1_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'ext1/fe2_2')
	stage4_tb_ext_pm4_b2a_relu = mx.sym.Activation(data = stage4_tb_ext_pm4_b2a_bn, act_type = 'relu', name = 'stage4_tb/ext/pm4/b2a/relu')
	stage4_tb_ext_pm3_b2c = mx.sym.Convolution(data=stage4_tb_ext_pm3_b2b_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm3/b2c')
	ext1_fe2_2_relu = mx.sym.Activation(data = ext1_fe2_2, act_type = 'relu', name = 'ext1/fe2_2/relu')
	stage4_tb_ext_pm4_b2b = mx.sym.Convolution(data=stage4_tb_ext_pm4_b2a_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm4/b2b')
	stage4_tb_ext_pm3_b2c_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3_b2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm3/b2c/bn')
	ext1_fe3_1	  = mx.sym.Convolution(data=ext1_fe2_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = False, layout = 'NCHW', name = 'ext1/fe3_1')
	stage4_tb_ext_pm5_b2a = mx.sym.Convolution(data=ext1_fe2_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm5/b2a')
	stage4_tb_ext_pm5 = mx.sym.Convolution(data=ext1_fe2_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm5')
	stage4_tb_ext_pm4_b2b_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm4_b2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm4/b2b/bn')
	stage4_tb_ext_pm3_res = mx.sym.broadcast_add(stage4_tb_ext_pm3_bn, stage4_tb_ext_pm3_b2c_bn)
	ext1_fe3_1_relu = mx.sym.Activation(data = ext1_fe3_1, act_type = 'relu', name = 'ext1/fe3_1/relu')
	stage4_tb_ext_pm5_b2a_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm5_b2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm5/b2a/bn')
	stage4_tb_ext_pm5_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm5, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm5/bn')
	stage4_tb_ext_pm4_b2b_relu = mx.sym.Activation(data = stage4_tb_ext_pm4_b2b_bn, act_type = 'relu', name = 'stage4_tb/ext/pm4/b2b/relu')
	stage4_tb_ext_pm3_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_res, act_type = 'relu', name = 'stage4_tb/ext/pm3/res/relu')
	ext1_fe3_2	  = mx.sym.Convolution(data=ext1_fe3_1_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = False, layout = 'NCHW', name = 'ext1/fe3_2')
	stage4_tb_ext_pm5_b2a_relu = mx.sym.Activation(data = stage4_tb_ext_pm5_b2a_bn, act_type = 'relu', name = 'stage4_tb/ext/pm5/b2a/relu')
	stage4_tb_ext_pm4_b2c = mx.sym.Convolution(data=stage4_tb_ext_pm4_b2b_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm4/b2c')
	ext1_fe3_2_relu = mx.sym.Activation(data = ext1_fe3_2, act_type = 'relu', name = 'ext1/fe3_2/relu')
	stage4_tb_ext_pm5_b2b = mx.sym.Convolution(data=stage4_tb_ext_pm5_b2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm5/b2b')
	stage4_tb_ext_pm4_b2c_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm4_b2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm4/b2c/bn')
	stage4_tb_ext_pm6_b2a = mx.sym.Convolution(data=ext1_fe3_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm6/b2a')
	stage4_tb_ext_pm6 = mx.sym.Convolution(data=ext1_fe3_2_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm6')
	stage4_tb_ext_pm5_b2b_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm5_b2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm5/b2b/bn')
	stage4_tb_ext_pm4_res = mx.sym.broadcast_add(stage4_tb_ext_pm4_bn, stage4_tb_ext_pm4_b2c_bn)
	stage4_tb_ext_pm6_b2a_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm6_b2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm6/b2a/bn')
	stage4_tb_ext_pm6_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm6, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm6/bn')
	stage4_tb_ext_pm5_b2b_relu = mx.sym.Activation(data = stage4_tb_ext_pm5_b2b_bn, act_type = 'relu', name = 'stage4_tb/ext/pm5/b2b/relu')
	stage4_tb_ext_pm4_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm4_res, act_type = 'relu', name = 'stage4_tb/ext/pm4/res/relu')
	stage4_tb_ext_pm6_b2a_relu = mx.sym.Activation(data = stage4_tb_ext_pm6_b2a_bn, act_type = 'relu', name = 'stage4_tb/ext/pm6/b2a/relu')
	stage4_tb_ext_pm5_b2c = mx.sym.Convolution(data=stage4_tb_ext_pm5_b2b_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm5/b2c')
	stage4_tb_ext_pm6_b2b = mx.sym.Convolution(data=stage4_tb_ext_pm6_b2a_relu, kernel=(3, 3), stride=(1, 1), dilate = (), pad=(1, 1), num_filter = 128, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm6/b2b')
	stage4_tb_ext_pm5_b2c_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm5_b2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm5/b2c/bn')
	stage4_tb_ext_pm6_b2b_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm6_b2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm6/b2b/bn')
	stage4_tb_ext_pm5_res = mx.sym.broadcast_add(stage4_tb_ext_pm5_bn, stage4_tb_ext_pm5_b2c_bn)
	stage4_tb_ext_pm6_b2b_relu = mx.sym.Activation(data = stage4_tb_ext_pm6_b2b_bn, act_type = 'relu', name = 'stage4_tb/ext/pm6/b2b/relu')
	stage4_tb_ext_pm5_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm5_res, act_type = 'relu', name = 'stage4_tb/ext/pm5/res/relu')
	stage4_tb_ext_pm6_b2c = mx.sym.Convolution(data=stage4_tb_ext_pm6_b2b_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm6/b2c')
	stage4_tb_ext_pm6_b2c_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm6_b2c, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm6/b2c/bn')
	stage4_tb_ext_pm6_res = mx.sym.broadcast_add(stage4_tb_ext_pm6_bn, stage4_tb_ext_pm6_b2c_bn)
	stage4_tb_ext_pm6_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm6_res, act_type = 'relu', name = 'stage4_tb/ext/pm6/res/relu')
	# from_layers = [stage4_tb_ext_pm2_res_relu, stage4_tb_ext_pm3_res_relu, stage4_tb_ext_pm4_res_relu, stage4_tb_ext_pm5_res_relu, stage4_tb_ext_pm6_res_relu]
	# sizes = [[.1,.15,.2], [.25,.3, .35], [.4,.45, .5], [.55, .6,.65], [.7, .75,.8]]
	# ratios = [[0.8,1,1.5,2], [0.8,1,1.5,2], [0.8,1,1.5,2],[0.8,1,1.5,2],[0.8,1,1.5,2]]
	
	stage4_tb_ext_pm4_feat_deconv_pre = mx.sym.Convolution(data=stage4_tb_ext_pm4_res_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4/tb/ext/pm4/feat/deconv/pre')	
	stage4_tb_ext_pm4_feat_deconv_pre_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm4_feat_deconv_pre, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4/tb/ext/pm4/feat/deconv/pre/bn')
	stage4_tb_ext_pm4_feat_deconv_pre_relu = mx.sym.Activation(data = stage4_tb_ext_pm4_feat_deconv_pre_bn, act_type = 'relu', name = 'stage4/tb/ext/pm4/feat/deconv/pre/relu')

	stage4_tb_ext_pm4_feat_deconv = mx.sym.Deconvolution(data=stage4_tb_ext_pm4_feat_deconv_pre_relu, kernel=(2, 2), stride=(2, 2), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4/tb/ext/pm4/feat/deconv')	
	stage4_tb_ext_pm4_feat_deconv_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm4_feat_deconv, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4/tb/ext/pm4/feat/deconv/bn')
	stage4_tb_ext_pm4_feat_deconv_relu = mx.sym.Activation(data = stage4_tb_ext_pm4_feat_deconv_bn, act_type = 'relu', name = 'stage4/tb/ext/pm4/feat/deconv/relu')	

	stage4_tb_ext_pm3_res_hyper = mx.sym.Convolution(data=stage4_tb_ext_pm3_res_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm3/res/hyper')	
	stage4_tb_ext_pm3_res_hyper_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3_res_hyper, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm3/res/hyper/bn')
	stage4_tb_ext_pm3_res_hyper_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_res_hyper_bn, act_type = 'relu', name = 'stage4_tb/ext/pm3/res/hyper/relu')

	stage4_tb_ext_pm3_feat = mx.sym.broadcast_add(stage4_tb_ext_pm3_res_hyper_relu, stage4_tb_ext_pm4_feat_deconv_relu)	
	stage4_tb_ext_pm3_feat_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_feat, act_type = 'relu', name = 'stage4/tb/ext/pm3/res/deconv/pre/relu')
		
	stage4_tb_ext_pm3_feat_deconv_pre = mx.sym.Convolution(data=stage4_tb_ext_pm3_feat_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4/tb/ext/pm3/feat/deconv/pre')	
	stage4_tb_ext_pm3_feat_deconv_pre_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3_feat_deconv_pre, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4/tb/ext/pm3/feat/deconv/pre/bn')
	stage4_tb_ext_pm3_feat_deconv_pre_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_feat_deconv_pre_bn, act_type = 'relu', name = 'stage4/tb/ext/pm3/feat/deconv/pre/relu')

	stage4_tb_ext_pm3_feat_deconv = mx.sym.Deconvolution(data=stage4_tb_ext_pm3_feat_deconv_pre_relu, kernel=(2, 2), stride=(2, 2), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4/tb/ext/pm3/feat/deconv')	
	stage4_tb_ext_pm3_feat_deconv_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm3_feat_deconv, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4/tb/ext/pm3/feat/deconv/bn')
	stage4_tb_ext_pm3_feat_deconv_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_feat_deconv_bn, act_type = 'relu', name = 'stage4/tb/ext/pm3/feat/deconv/relu')	

	stage4_tb_ext_pm2_res_hyper = mx.sym.Convolution(data=stage4_tb_ext_pm2_res_relu, kernel=(1, 1), stride=(1, 1), dilate = (), pad=(0, 0), num_filter = 256, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage4_tb/ext/pm2/res/hyper')	
	stage4_tb_ext_pm2_res_hyper_bn = mx.sym.BatchNorm(data = stage4_tb_ext_pm2_res_hyper, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, use_global_stats = False, name = 'stage4_tb/ext/pm2/res/hyper/bn')
	stage4_tb_ext_pm2_res_hyper_relu = mx.sym.Activation(data = stage4_tb_ext_pm2_res_hyper_bn, act_type = 'relu', name = 'stage4_tb/ext/pm2/res/hyper/relu')

	stage4_tb_ext_pm2_feat = mx.sym.broadcast_add(stage4_tb_ext_pm2_res_hyper_relu, stage4_tb_ext_pm3_feat_deconv_relu)	
	stage4_tb_ext_pm2_feat_relu = mx.sym.Activation(data = stage4_tb_ext_pm2_feat, act_type = 'relu', name = 'stage4/tb/ext/pm2/res/deconv/pre/relu')
	

	from_layers = [stage4_tb_ext_pm2_feat_relu, stage4_tb_ext_pm3_feat_relu, stage4_tb_ext_pm4_res_relu]
	sizes = [[0.1,0.16, 0.22], [0.3,0.38, 0.46], [0.56,0.66,0.76]]
	ratios = [[0.25, 0.5, 1.0],[0.25, 0.5, 1.0], [0.33,0.5,1.0]]
	
	normalizations = [-1, -1, -1]
	steps = []
	num_channels = [-1, -1, -1]
	loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
		num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
		num_channels=num_channels, clip=False, interm_layer=0, steps=steps)

	tmp = mx.symbol.contrib.MultiBoxTarget(
		*[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
		ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
		negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
		name="multibox_target")
	loc_target = tmp[0]
	loc_target_mask = tmp[1]
	cls_target = tmp[2]

	cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
		ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
		normalization='valid', name="cls_prob")
	loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
		data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
	loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
		normalization='valid', name="loc_loss")

	cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
	det = mx.symbol.contrib.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
		name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
		variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
	det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

	out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
	return out	
	

def get_symbol(num_classes=7, nms_thresh=0.5, force_suppress=False,
			   nms_topk=400, **kwargs):
	net = get_symbol_train(num_classes)
	cls_preds = net.get_internals()["multibox_cls_pred_output"]
	loc_preds = net.get_internals()["multibox_loc_pred_output"]
	anchor_boxes = net.get_internals()["multibox_anchors_output"]

	cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
		name='cls_prob')
	out = mx.symbol.contrib.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
		name="detection", nms_threshold=nms_thresh, force_suppress=True,
		variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
	return out		
	
def deploy_weight(model, weight_file):

	if weight_file == None:
		return

	try:
		weights_dict = np.load(weight_file).item()
	except:
		weights_dict = np.load(weight_file, encoding='bytes').item()

	arg_params = dict()
	aux_params = dict()
	for weight_name, weight_data in weights_dict.items():
		weight_name = str(weight_name)
		if "moving" in weight_name:
			aux_params[weight_name] = mx.nd.array(weight_data)
		else:
			arg_params[weight_name] = mx.nd.array(weight_data)

	model.bind(for_training = False, data_shapes = [('data', (1, 3, 305, 305))])
	model.set_params(arg_params = arg_params, aux_params = aux_params, allow_missing = True)

	return model


from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


def get_image(url, show=False):
	import cv2
	# download and show the image
	fname = mx.test_utils.download(url)
	img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
	if img is None:
		return None
	if show:
		import matplotlib.pyplot as plt
		plt.imshow(img)
		plt.axis('off')
	# convert into format (batch, RGB, width, height)
	img = cv2.resize(img, (224, 224))
	img = np.swapaxes(img, 0, 2)
	img = np.swapaxes(img, 1, 2)
	img = img[np.newaxis, :]
	return img


def predict(model, labels, url):
	# to show the image, change the argument show into True
	img = get_image(url, show = False)
	# compute the predict probabilities
	model.forward(Batch([mx.nd.array(img)]))
	prob = model.get_outputs()[0].asnumpy()
	# print the top-5
	prob = np.squeeze(prob)
	a = np.argsort(prob)[::-1]
	for i in a[0:5]:
		print('prbability = %f, class = %s' %(prob[i], labels[i]))


if __name__ == '__main__':
	model = RefactorModel()
	# remember to adjust params path
	model = deploy_weight(model, 'mxnet_mxnet_peleenet_v2-0000.params')

	# # call function predict
	# with open('synset.txt', 'r') as f:
	#	 labels = [l.rstrip() for l in f]
	# predict(model, labels, 'http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')
