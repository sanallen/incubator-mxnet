# --------------------------------------------------------
# Focal loss
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by unsky https://github.com/unsky/
# --------------------------------------------------------

"""
Focal loss 
"""

import mxnet as mx
import numpy as np
from mxnet import autograd
class FocalLossOperator(mx.operator.CustomOp):
	def __init__(self,  gamma, alpha,use_ignore,ignore_label,num_classes):
		super(FocalLossOperator, self).__init__()
		self._gamma = gamma
		self._alpha = alpha 
		self.use_ignore = use_ignore
		self.ignore_label = ignore_label
		self.eps = 1e-20
		self.num_classes = num_classes
	def forward(self, is_train, req, in_data, out_data, aux):
		cls_score = in_data[0][:]
		# print(in_data[0].shape)
		# pro_ =(mx.nd.sigmoid(cls_score)+self.eps).asnumpy()
		pro_ = (mx.nd.SoftmaxActivation( cls_score,mode="channel")).asnumpy()

		
		

		self.assign(out_data[0],req[0],mx.nd.array(pro_))
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
	   
		label = in_data[1].asnumpy()[:].astype('int')[:]
		# print(in_data[1].asnumpy().shape,in_data[0].shape)
		data = in_data[0][:]
		# focal loss onehot
		# label[label<0] = 0
		labels_ = np.zeros((label.shape[0], self.num_classes + 1, label.shape[1]))
		ig_cnt = 0
		pos_cnt = 0
		for i in range(label.shape[0]):
			labels_[i, label[i], np.arange(label.shape[1],dtype = 'int') ] = 1
			# label_ig = in_data[1].asnumpy()[:].astype('int')[:].flatten()[:]
			ind_ig = np.where(label[i]<0)[0]
			
			pos_cnt += len(np.where(label[i]>0)[0])
			ig_cnt += len(ind_ig)
			
			labels_[i, :, ind_ig] = 0

		# print labels_.shape, data.shape
		labels_ = mx.nd.array(labels_)	
		

		# label_ig = in_data[1].asnumpy()[:].astype('int')[:].flatten()[:] - 1

	  #  aaa = np.where(label_ig>=0)[0]
		# ind_ig = np.where(label_ig<-1)[0]
		# print(cnt)
		nom = label.shape[0]*label.shape[1] - ig_cnt

		# print('nom,',nom)

		data.attach_grad()
		with autograd.record():
			# pro = mx.nd.sigmoid(data)
		
			# print pro.asnumpy().shape
			pro = mx.nd.SoftmaxActivation(data,mode="channel")
			# print pro.asnumpy().shape
			focal_loss = mx.nd.sum(-1*labels_*mx.nd.power(1 -  pro, self._gamma)*mx.nd.log(pro + self.eps)) / pos_cnt
			# focal_loss =  mx.nd.sum(-1 * labels_ *self._alpha * mx.nd.power(1 -  pro + self.eps, self._gamma) * mx.nd.log( pro+self.eps) - (1-self._alpha) * (1-labels_) * mx.nd.power(pro + self.eps, self._gamma) * mx.nd.log(1 - pro + self.eps))/pos_cnt
			focal_loss.backward()
		grad = data.grad.asnumpy()
		
		# print(grad.shape)

		# grad[0,:,ind_ig] = 0
		# g = np.sum(grad.flatten()[grad.flatten()>0])
		# n = np.sum(grad>0)
		# print n,'cls: ',g/n
	  
		self.assign(in_grad[0], req[0], mx.nd.array(grad))
		# self.assign(in_grad[1],req[1],0)

@mx.operator.register('FocalLoss')
class FocalLossProp(mx.operator.CustomOpProp):
	def __init__(self, num_classes,gamma,alpha,use_ignore,ignore_label):
		super(FocalLossProp, self).__init__(need_top_grad=False)
		self.use_ignore =bool(use_ignore)
		
		self.ignore_label = int(ignore_label)
		self.num_classes = int(num_classes)
		self._gamma = float(gamma)
		self._alpha = float(alpha)

	def list_arguments(self):
		return ['data', 'labels']

	def list_outputs(self):
		return ['focal_loss']

	def infer_shape(self, in_shape):
		data_shape = in_shape[0]
		labels_shape = in_shape[1]
		out_shape = data_shape

		return  [data_shape, labels_shape],[out_shape]

	def create_operator(self, ctx, shapes, dtypes):
		return FocalLossOperator(self._gamma,self._alpha,self.use_ignore,self.ignore_label,self.num_classes)

   
