import mxnet as mx
import numpy as np

def pelee_unit(data, filter_num_branch1a, filter_num_branch1b, stride, name, bn_mom = 0.0):
   
    """Return PeleeNet Unit symbol for building PeleeNet
    Parameters
    ----------
    data : mxnet.symbol.symbol.Symbol
        Input data
    filter_num_branch1a : list
        Channel size of  branch1a
    filter_num_branch1b : list
        Channel size of  branch1b
    stride : tuple
        Stride used in convolution
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """

    branch1a = mx.sym.Convolution(data = data, kernel = (1L, 1L), stride = (1L, 1L), dilate = (1, 1), pad = (0L, 0L), 
        num_filter = filter_num_branch1a, num_group = 1, no_bias = True, layout = 'NCHW', name = name + '/branch1a')
    branch1a_bn = mx.sym.BatchNorm(data = branch1a, axis = 1, eps = 0.0010000000475, momentum = bn_mom, 
        fix_gamma = False, use_global_stats = False, name = name + '/branch1a/bn')
    branch1a_relu = mx.sym.Activation(data = branch1a_bn, act_type = 'relu', name = name + '/branch1a/relu')

    branch1b = mx.sym.Convolution(data = branch1a_relu, kernel = (3L, 3L), stride = (1L, 1L), dilate = (1, 1), 
        pad=(1L, 1L), num_filter = filter_num_branch1b, num_group = 1, no_bias = True, layout = 'NCHW', name = name+'/branch1b')
    branch1b_bn = mx.sym.BatchNorm(data = branch1b, axis = 1, eps = 0.0010000000475, momentum = bn_mom, 
        fix_gamma = False, use_global_stats = False, name = name + '/branch1b/bn')
    branch1b_relu = mx.sym.Activation(data = branch1b_bn, act_type = 'relu', name = name+'/branch1b/relu')

    concat = mx.sym.concat(data, branch1b_relu, dim = 1, name = name+'/concat')

    return concat


def stem(data):
    # define stem stage (stage 0)

    stem1 = mx.sym.Convolution(data=data, kernel=(3L, 3L), stride=(2L, 2L), dilate = (1, 1), pad=(1L, 1L), num_filter = 32, 
        num_group = 1, no_bias = True, layout = 'NCHW', name = 'stem1')
    stem1_bn = mx.sym.BatchNorm(data = stem1, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, 
        use_global_stats = False, name = 'stem1/bn')
    stem1_relu = mx.sym.Activation(data = stem1_bn, act_type = 'relu', name = 'stem1/relu')
    
    stem2_pool = mx.sym.Pooling(data = stem1_relu, global_pool = False, kernel=(2L, 2L), pool_type = 'max', 
        stride=(2L, 2L), name = 'stem/pool')
    stem2a = mx.sym.Convolution(data=stem1_relu, kernel=(1L, 1L), stride=(1L, 1L), dilate = (1, 1), pad=(0L, 0L), 
        num_filter = 16, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stem2a')
    stem2a_bn = mx.sym.BatchNorm(data = stem2a, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, 
        use_global_stats = False, name = 'stem2a/bn')
    stem2a_relu = mx.sym.Activation(data = stem2a_bn, act_type = 'relu', name = 'stem2a/relu')
    stem2b = mx.sym.Convolution(data=stem2a_relu, kernel=(3L, 3L), stride=(2L, 2L), dilate = (1, 1), pad=(1L, 1L), 
        num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stem2b')
    stem2b_bn = mx.sym.BatchNorm(data = stem2b, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, 
        use_global_stats = False, name = 'stem2b/bn')
    stem2b_relu = mx.sym.Activation(data = stem2b_bn, act_type = 'relu', name = 'stem2b/relu')
    stem2_concat = mx.sym.concat(stem2_pool, stem2b_relu, dim = 1, name = 'stem/concat')

    stem3 = mx.sym.Convolution(data=stem2_concat, kernel=(1L, 1L), stride=(1L, 1L), dilate = (1, 1), pad=(0L, 0L), 
        num_filter = 32, num_group = 1, no_bias = True, layout = 'NCHW', name = 'stem3')
    stem3_bn = mx.sym.BatchNorm(data = stem3, axis = 1, eps = 0.0010000000475, momentum = 0.0, fix_gamma = False, 
        use_global_stats = False, name = 'stem3/bn')
    stem3_relu = mx.sym.Activation(data = stem3_bn, act_type = 'relu', name = 'stem3/relu') 

    return stem3_relu

def peleenet(units, num_stages, filter_list_branch1a, filter_list_branch1b, filter_list_transition, num_classes, 
    image_shape, bn_mom = 0.0, dtype = 'float32'):

    """Return PeleeNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list_branch1a : list
        Channel size of each stage's branch1a (except stage0)
    filter_list_branch1b : list
        Channel size of each stage's branch1b (except stage0)
    num_classes : int
        Ouput size of symbol
    image_shape : tuple
        Input size of image
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """  

    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    # create the stem stage
    stem3_relu = stem(data) # creat the stem stage 

    # create stage1 to stage4
    stage_tb_pool = stem3_relu # initialize dense layer
    for i in range(num_stages):
        dense_layer = stage_tb_pool
        for j in range(units[i]):
            dense_layer = pelee_unit(data = dense_layer, filter_num_branch1a = filter_list_branch1a[i], 
                filter_num_branch1b = filter_list_branch1b[i], stride = (1L, 1L), name = 'stage%d_%d' % (i+1, j+1))

        stage_tb = mx.sym.Convolution(data=dense_layer, kernel=(1L, 1L), stride=(1L, 1L), dilate = (1, 1), 
            pad=(0L, 0L), num_filter = filter_list_transition[i], num_group = 1, no_bias = True, layout = 'NCHW', name = 'stage%d_tb' % (i+1))
        stage_tb_bn = mx.sym.BatchNorm(data = stage_tb, axis = 1, eps = 0.0010000000475, momentum = 0.0, 
            fix_gamma = False, use_global_stats = False, name = 'stage%d_tb/bn' % (i+1))
        stage_tb_relu = mx.sym.Activation(data = stage_tb_bn, act_type = 'relu', name = 'stage%d_tb/relu' % (i+1))
        stage_tb_pool  = mx.sym.Pooling(data = stage_tb_relu, global_pool = False, kernel=(2L, 2L), pool_type = 'avg', 
            stride=(2L, 2L), name = 'stage%d_tb/pool' % (i+1))

    # create the classification layer
    global_pool = mx.sym.Pooling(data = stage_tb_relu, global_pool = False, kernel=(7L, 7L), pool_type = 'avg', 
        stride=(1L, 1L), pad=(0L, 0L), name = 'global_pool')
    classifier_0 = mx.sym.flatten(data = global_pool, name = 'classifier_0')
    classifier_1 = mx.sym.FullyConnected(data = classifier_0, num_hidden = 3341, no_bias = False, name = 'classifier_1')

    return mx.sym.SoftmaxOutput(data=classifier_1, name='softmax')




def get_symbol(num_classes, image_shape, dtype = 'float32', **kwargs): 

    """Return ResNet symbol of
    Parameters
    ----------
    num_classes : int
        Ouput size of symbol
    image_shape : tuple
        Input size of image
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """ 

    image_shape = [int(l) for l in image_shape.split(',')]

    units = [3,4,8,6]
    num_stages = 4
    filter_list_branch1a = [16, 32, 64, 64]
    filter_list_branch1b = [32, 32, 32, 32]
    filter_list_transition = [128, 256, 512, 704]

    return peleenet(units = units, num_stages = num_stages, 
        filter_list_branch1a = filter_list_branch1a, filter_list_branch1b = filter_list_branch1b, 
        filter_list_transition = filter_list_transition, num_classes = num_classes, image_shape = image_shape, dtype = dtype)
