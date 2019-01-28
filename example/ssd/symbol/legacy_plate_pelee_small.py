import mxnet as mx
import numpy as np
import math
from common import legacy_conv_act_layer
from common import multibox_layer
import matplotlib.pyplot as plt

def _conv_block(data, num_output, kernel_size, stride, pad, name):
	conv = mx.sym.Convolution(data=data, kernel=(kernel_size, kernel_size), stride=(stride, stride), dilate = (1, 1), pad=(pad, pad), \
					num_filter = num_output, num_group = 1, no_bias = True, layout = 'NCHW', name = name)
	conv_bn = mx.sym.BatchNorm(data = conv, axis = 1, eps = 0.0010000000475, \
					momentum = 0.0, fix_gamma = False, use_global_stats = False, name = '{}/bn'.format(name))
	conv_relu = mx.sym.Activation(data = conv_bn, act_type = 'relu', name = '{}/relu'.format(name))

	return conv_relu
	
def _deconv_block(data, num_output, kernel_size, stride, pad, name):
	conv = mx.sym.Deconvolution(data=data, kernel=(kernel_size, kernel_size), stride=(stride, stride), dilate = (1, 1), pad=(pad, pad), \
					num_filter = num_output, num_group = 1, no_bias = True, layout = 'NCHW', name = name)
	conv_bn = mx.sym.BatchNorm(data = conv, axis = 1, eps = 0.0010000000475, \
					momentum = 0.0, fix_gamma = False, use_global_stats = False, name = '{}/bn'.format(name))
	conv_relu = mx.sym.Activation(data = conv_bn, act_type = 'relu', name = '{}/relu'.format(name))

	return conv_relu

def _dense_block(data, num_layers, growth_rate,bottleneck_width,name):
	x = data
	growth_rate = int(growth_rate/2)
	
	for i in range(num_layers):
		base_name = '{}_{}'.format(name,i+1)
		
		inter_channel = int(growth_rate * bottleneck_width / 4) * 4
		
		cb2 = _conv_block(x, inter_channel, 1, 1, 0, '{}/branch2a'.format(base_name))
		cb2 = _conv_block(cb2, growth_rate * 2, 3, 1, 1, '{}/branch2b'.format(base_name))
		
		x = mx.sym.concat(x, cb2, dim = 1, name = '{}/concat'.format(base_name))
	
	return x
	
def _transition_block(data, num_filter, with_pooling, name):
	x = _conv_block(data, num_filter, 1, 1, 0, name)
	if with_pooling:
		pool_name = '{}/pool'.format(name)
		x = mx.sym.Pooling(data = x, global_pool = False, kernel=(2, 2), pool_type = 'avg', stride=(2, 2), name = pool_name)
	return x

def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=True,
					 nms_topk=400, **kwargs):
										 
	block_config = [2,4,4]
	bottleneck_width = [1,2,4]
	growth_rate = [32,32,32]
	num_init_features = 64
	total_filters = [128, 256, 448]
	
	total_filter = num_init_features
	if type(bottleneck_width) is list:
		bottleneck_widths = bottleneck_width
	else:
		bottleneck_widths = [bottleneck_width] * 4
	if type(growth_rate) is list:
		growth_rates = growth_rate
	else:
		growth_rates = [growth_rate] * 4		
		
#######################################################
					 
	data			= mx.sym.var('data')
	label = mx.symbol.Variable(name="label")	

	# stem1 = _conv_block(data, 16, 3, 2, 1, 'stem1')
	# stem2 = _conv_block(stem1, 32, 3, 2, 1, 'stem2')
	# stem3 = _conv_block(stem2, 64, 3, 2, 1, 'stem3')
	
	stem1 = _conv_block(data, 16, 3, 2, 1, 'stem1')
	stem2 = _conv_block(stem1, 32, 3, 1, 1, 'stem2')
	stem3 = _conv_block(stem2, 64, 3, 1, 1, 'stem3')

	from_layer = stem3
	# from_layer = data
	
	feat_layers = []
	
	for idx, num_layers in enumerate(block_config):
		from_layer = _dense_block(from_layer, num_layers, growth_rates[idx], bottleneck_widths[idx],'stage{}'.format(idx+1))
		# total_filter += growth_rates[idx] * num_layers
		total_filter = total_filters[idx]
		if idx == len(block_config) - 1:
			with_pooling=False
		else:
			with_pooling=True
 
		from_layer = _transition_block(from_layer, total_filter, with_pooling, 'stage{}_tb'.format(idx+1))
		if idx >= 1:
			feat_layers.append(from_layer)
#######################################################		
	# print from_layer.get_int()
	stage2_tb = from_layer.get_internals()['stage2_tb/relu_output']
	
	stage4_tb_ext_pm2 = _conv_block(stage2_tb, 256, 1, 1, 0, 'stage4_tb_ext_pm2')	
	stage4_tb_ext_pm2_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm2, act_type = 'relu', name = 'stage4_tb_ext_pm2_res/relu')
	
	stage3_tb = from_layer.get_internals()['stage3_tb/relu_output']
	stage4_tb_ext_pm3 = _conv_block(stage3_tb, 256, 1, 1, 0, 'stage4_tb_ext_pm3')
	
	stage4_tb_ext_pm3_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm3, act_type = 'relu', name = 'stage4_tb_ext_pm3_res/relu')
	
	stage4_tb_relu_ext1_fe1_1 = _conv_block(stage3_tb, 256, 1, 1, 0, 'stage4_tb_relu_ext1_fe1_1')
	ext1_fe1_2 = _conv_block(stage4_tb_relu_ext1_fe1_1, 256, 3, 2, 1, 'ext1_fe1_2')
	stage4_tb_ext_pm4 = _conv_block(ext1_fe1_2, 256, 1, 1, 0, 'stage4_tb_ext_pm4')
	
	stage4_tb_ext_pm4_res_relu = mx.sym.Activation(data = stage4_tb_ext_pm4, act_type = 'relu', name = 'stage4_tb_ext_pm4_res/relu')
	
#######################################################

	stage4_tb_ext_pm4_feat_deconv_pre = _conv_block(stage4_tb_ext_pm4_res_relu, 256, 1, 1, 0, 'stage4/tb/ext/pm4/feat/deconv/pre')	
	stage4_tb_ext_pm4_feat_deconv = _deconv_block(stage4_tb_ext_pm4_feat_deconv_pre, 256, 2, 2, 0, 'stage4/tb/ext/pm4/feat/deconv')		
	stage4_tb_ext_pm3_res_hyper = _conv_block(stage4_tb_ext_pm3_res_relu, 256, 1, 1, 0, 'stage4_tb/ext/pm3/res/hyper/relu')

	stage4_tb_ext_pm3_feat = mx.sym.broadcast_add(stage4_tb_ext_pm3_res_hyper, stage4_tb_ext_pm4_feat_deconv)	
	stage4_tb_ext_pm3_feat_relu = mx.sym.Activation(data = stage4_tb_ext_pm3_feat, act_type = 'relu', name = 'stage4/tb/ext/pm3/res/deconv/pre/relu')
		
	stage4_tb_ext_pm3_feat_deconv_pre = _conv_block(stage4_tb_ext_pm3_feat_relu, 256, 1, 1, 0, 'stage4/tb/ext/pm3/feat/deconv/pre')	
	stage4_tb_ext_pm3_feat_deconv = _deconv_block(stage4_tb_ext_pm3_feat_deconv_pre, 256, 2, 2, 0, 'stage4/tb/ext/pm3/feat/deconv')	

	stage4_tb_ext_pm2_res_hyper = _conv_block(stage4_tb_ext_pm2_res_relu, 256, 1, 1, 0, 'stage4_tb/ext/pm2/res/hyper/relu')	
	
	stage4_tb_ext_pm2_feat = mx.sym.broadcast_add(stage4_tb_ext_pm2_res_hyper, stage4_tb_ext_pm3_feat_deconv)	
	stage4_tb_ext_pm2_feat_relu = mx.sym.Activation(data = stage4_tb_ext_pm2_feat, act_type = 'relu', name = 'stage4/tb/ext/pm2/res/deconv/pre/relu')

#######################################################

	from_layers = [stage4_tb_ext_pm2_feat_relu, stage4_tb_ext_pm3_feat_relu, stage4_tb_ext_pm4_res_relu]
	# sizes = [[0.1,0.16, 0.22], [0.3,0.38, 0.46], [0.56,0.66,0.76]]
	# ratios = [[0.25, 0.5, 1.0],[0.25, 0.5, 1.0,1.5], [0.33,0.5,1.0,1.5]]
	sizes = [[0.05, 0.1, 0.16], [0.18, 0.2, 0.22], [0.25, 0.28, 0.3]]
	ratios = [[0.8, 1, 3.14],[1, 2, 3.14, 3.42], [1.5, 2, 3.14, 3.6]]  
	
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
	
def get_symbol(num_classes=7, nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
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
	
	
if __name__ == '__main__':
	# net = get_symbol(args.network, args.data_shape,
    #     num_classes=args.num_classes, nms_thresh=args.nms_thresh,
    #     force_suppress=args.force_nms, nms_topk=args.nms_topk)
	net = get_symbol(1, 0.45, True, 400)
	mx.viz.plot_network(net, shape={"data":(1, 3, 320, 320), }, node_attrs={"shape":'rect',"fixedsize":'false'}).render("legacy_plate_pelee_320")
	