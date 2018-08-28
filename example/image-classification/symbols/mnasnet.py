import mxnet as mx
import numpy as np

def shortcut(data_in, data_residual, prefix):
    out=mx.sym.elemwise_add(data_in, data_residual, name='%s_elemwist_add'%prefix)
    return out

def Conv(data, num_filter, kernel, stride, pad, prefix='conv'):
    conv = mx.sym.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, 
        pad = pad, name = prefix+ '/conv', dilate = (1, 1))
    bn = mx.sym.BatchNorm(data = conv, name = prefix + '/bn')
    act = mx.sym.Activation(data = bn, act_type = 'relu', name = prefix + '/relu')
    return act

def Conv1x1(data, num_filter, stride, if_act = False, prefix = 'conv1x1'):

    conv = mx.sym.Convolution(data = data, num_filter = num_filter, kernel = (1, 1), stride = stride, 
        pad = (0, 0), name = prefix+ '/conv', dilate = (1, 1))
    bn = mx.sym.BatchNorm(data = conv, name = prefix + '/bn')
    if if_act:
        act = mx.sym.Activation(data = bn, act_type = 'relu', name = prefix + '/relu')
        return act
    return bn

def Depthwise_conv(data, num_filter, kernel, stride, pad, prefix = 'depthwise_conv'):
    conv = mx.sym.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, 
        pad = pad, name = prefix+ '/conv', dilate = (1, 1), num_group = num_filter)
    bn = mx.sym.BatchNorm(data = conv, name = prefix + '/bn')
    act = mx.sym.Activation(data = bn, act_type = 'relu', name = prefix + '/relu')
    return act

def Sep_conv(data, num_in_filter, num_out_filter, kernel, stride, pad, prefix = 'sep_conv'):
    dwise_conv = Depthwise_conv(data = data, num_filter = num_in_filter, kernel = kernel, stride = stride, pad = pad, prefix = prefix+'_dwise_conv')
    conv1x1 = Conv1x1(data = dwise_conv, num_filter = num_out_filter, stride = (1, 1), if_act = True, prefix = prefix+'_conv1x1')
    return conv1x1

def Inverted_residual_unit(data, num_in_filter, num_out_filter, ifshortcut, stride, kernel, pad, expansion_factor, prefix):
    if ifshortcut:
        assert (num_in_filter == num_out_filter) , "the channels of element add inputs should be equal"   
    num_expfilter = int(round(num_in_filter*expansion_factor))

    channel_expand = Conv1x1(
        data=data,
        num_filter=num_expfilter,
        stride=(1,1),
        if_act=True,
        prefix='%s_exp'%prefix,
    )
    bottleneck_conv = Depthwise_conv(
        data= channel_expand,
        num_filter=num_expfilter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        prefix='%s_depthwise'%prefix,
    )
    linear_out = Conv1x1(
        data=bottleneck_conv,
        num_filter=num_out_filter,
        stride=(1, 1),
        if_act=False,
        prefix='%s_linear'%prefix
    )
    if ifshortcut:
        out = shortcut(
            data_in=data,
            data_residual=linear_out,
            prefix=prefix,
        ) 
        return out
    else:
        return linear_out

def MBconv(data, num_in_filter, expansion_factor, num_out_filter, repeats, stride, kernel, pad, ifshortcut, prefix):
    # if ifshortcut:
    #     assert (num_in_filter == num_out_filter) , "the channels of element add inputs should be equal"

    if (not isinstance(stride, list)) and  (not isinstance(stride, tuple)):
        stride = (stride, stride)
    if (not isinstance(kernel, list)) and  (not isinstance(kernel, tuple)):
        kernel = (kernel, kernel)
    if (not isinstance(pad, list)) and  (not isinstance(pad, tuple)):
        pad = (pad, pad)
    
    first_block = Inverted_residual_unit(
        data=data,
        num_in_filter=num_in_filter,
        num_out_filter=num_out_filter,
        ifshortcut=False,
        stride=stride,  
        kernel=kernel,
        pad=pad,
        expansion_factor=expansion_factor,
        prefix='%s_block0'%prefix
    )

    last_residual_block = first_block

    for i in range(1,repeats):
        last_residual_block = Inverted_residual_unit(
            data=last_residual_block,
            num_in_filter=num_out_filter,
            num_out_filter=num_out_filter,
            ifshortcut=ifshortcut,
            stride=(1,1),  # all block have stride 1 except first block
            kernel=kernel,
            pad=pad,
            expansion_factor=expansion_factor,
            prefix='%s_block%d'%(prefix, i)
        )
    return last_residual_block

def Mnasnet(num_classes = 3341, **kwargs):

    first_num_filter = 32
    sep_num_in_filter = 32
    sep_num_out_filter = 16
    inverted_residual_setting = [
            # t, c,  n, s, k
            # expansion_factor, num_out_filter, repeats, stride, kernel, prefix
            [3, 24,  3, 2, 3, "stage2"],  # -> 56x56
            [3, 40,  3, 2, 5, "stage3"],  # -> 28x28
            [6, 80,  3, 2, 5, "stage4_1"],  # -> 14x14
            [6, 96,  2, 1, 3, "stage4_2"],  # -> 14x14
            [6, 192, 4, 2, 5, "stage5_1"], # -> 7x7
            # [6, 320, 1, 1, 3, "stage5_2"], # -> 7x7          
        ]
    last_num_filter = 1280

    data = mx.sym.Variable(name='data')
    first_conv = Conv(data, first_num_filter, kernel = (3, 3), stride = (2, 2), pad = (1, 1), prefix = 'stage1_conv')
    sep_conv = Sep_conv(data= first_conv, num_in_filter = sep_num_in_filter, num_out_filter = sep_num_out_filter, 
        kernel = (3,3), stride = (1, 1), pad = (1, 1), prefix = 'stage1')
    
    mb_conv = sep_conv
    num_in_filter = sep_num_out_filter
    for (t, c, n, s, k, p) in inverted_residual_setting:
        mb_conv = MBconv(data=mb_conv, 
            num_in_filter=num_in_filter, 
            expansion_factor=t, 
            num_out_filter=c, 
            repeats=n, 
            stride=s, 
            kernel=k,
            pad=k//2,
            ifshortcut=True, 
            prefix=p)
        num_in_filter = c
    # last stage has no shortcut
    mb_conv = MBconv(data=mb_conv, 
        num_in_filter=num_in_filter, 
        expansion_factor=6, 
        num_out_filter=320, 
        repeats=1, 
        stride=1, 
        kernel=3,
        pad=1,
        ifshortcut=False,  
        prefix="stage5_2")

    global_pool = mx.sym.Pooling(data = mb_conv, global_pool = True, kernel=(7, 7), pool_type = 'avg', 
        stride=(1, 1), pad=(0, 0), name = 'global_pool')
    flatten = mx.sym.flatten(data = global_pool, name = 'flatten')
    fc = mx.sym.FullyConnected(data = flatten, num_hidden = 3341, no_bias = False, name = 'fc')
    
    return mx.sym.SoftmaxOutput(data=fc, name='softmax')

def get_symbol(num_classes, **kwargs):
    return Mnasnet(num_classes = num_classes)

if __name__ == '__main__':
    net = Mnasnet(3341, prefix="")
    print net.get_internals().list_outputs()

