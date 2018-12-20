import mxnet as mx
import numpy as np

def Conv(data, num_filter = 1, kernel=(1, 1), stride = (1, 1), pad = (0, 0),  
    no_bias = True, layout = 'NCHW', name_conv = 'conv', axis = 1, eps = 0.001, momentum = 0.0, fix_gamma = False, 
    use_global_stats = False, act_type = 'relu'):
    conv = mx.sym.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, 
        pad = pad, no_bias = no_bias, layout = layout, name = name_conv+ '/conv', dilate = (1, 1))
    bn = mx.sym.BatchNorm(data = conv, axis = axis, eps = eps, momentum = momentum, fix_gamma = fix_gamma, 
        use_global_stats = use_global_stats, name = name_conv + '/bn')
    act = mx.sym.Activation(data = bn, act_type = act_type, name = name_conv + '/' + act_type)
    return act

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
    branch1a_relu = Conv(data = data, num_filter = filter_num_branch1a, kernel = (1, 1), stride = (1, 1), 
        pad = (0, 0), name_conv = name + '/branch1a')
    branch1b_relu = Conv(data = branch1a_relu, num_filter = filter_num_branch1b, kernel = (3, 3), stride = (1, 1), 
        pad = (1, 1), name_conv = name + '/branch1b')
    concat = mx.sym.concat(data, branch1b_relu, dim = 1, name = name+'/concat')
    return concat

def stem(data):
    # define stem stage (stage 0)
    stem1_relu = Conv(data = data, num_filter = 32, kernel = (3, 3), stride = (2, 2), pad = (1, 1), name_conv = 'stem1')
    stem2_pool = mx.sym.Pooling(data = stem1_relu, global_pool = False, kernel=(2, 2), pool_type = 'max', stride = (2, 2), name = 'stem/pool')
    stem2a_relu = Conv(data = stem1_relu, num_filter = 16, kernel = (1, 1), stride = (1, 1), pad = (0, 0), name_conv = 'stem2a', )     
    stem2b_relu = Conv(data = stem2a_relu, num_filter = 32, kernel = (3, 3), stride = (2, 2), pad = (1, 1), name_conv = 'stem2b') 
    stem2_concat = mx.sym.concat(stem2_pool, stem2b_relu, dim = 1, name = 'stem/concat')
    stem3_relu = Conv(data = stem2_concat, num_filter = 32, kernel = (1, 1), stride = (1, 1), pad = (0, 0), name_conv = 'stem3') 
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

    # create the stem stage
    stem3_relu = stem(data) # creat the stem stage 

    # create stage1 to stage4
    stage_tb_pool = stem3_relu # initialize dense layer
    for i in range(num_stages):
        dense_layer = stage_tb_pool
        for j in range(units[i]):
            dense_layer = pelee_unit(data = dense_layer, filter_num_branch1a = filter_list_branch1a[i], 
                filter_num_branch1b = filter_list_branch1b[i], stride = (1, 1), name = 'stage%d_%d' % (i+1, j+1))
        stage_tb_relu = Conv(data = dense_layer, num_filter = filter_list_transition[i], kernel = (1, 1), stride = (1, 1), 
            pad = (0, 0), name_conv = 'stage%d_tb' % (i+1))
        stage_tb_pool  = mx.sym.Pooling(data = stage_tb_relu, global_pool = False, kernel=(2, 2), pool_type = 'avg', 
            stride=(2, 2), name = 'stage%d_tb/pool' % (i+1))

    # create the classification layer
    global_pool = mx.sym.Pooling(data = stage_tb_relu, global_pool = True, kernel=(7, 7), pool_type = 'avg', 
        stride=(1, 1), pad=(0, 0), name = 'global_pool')
    classifier_0 = mx.sym.flatten(data = global_pool, name = 'classifier_0')
    classifier_1 = mx.sym.FullyConnected(data = classifier_0, num_hidden = num_classes, no_bias = False, name = 'classifier_1')

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

    units = [3, 4, 8, 6]
    num_stages = 4
    filter_list_branch1a = [16, 32, 64, 64]
    filter_list_branch1b = [32, 32, 32, 32]
    filter_list_transition = [128, 256, 512, 704]

    return peleenet(units = units, num_stages = num_stages, 
        filter_list_branch1a = filter_list_branch1a, 
        filter_list_branch1b = filter_list_branch1b, 
        filter_list_transition = filter_list_transition, 
        num_classes = num_classes, image_shape = image_shape, dtype = dtype)
