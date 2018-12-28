# delete the concat layer
import mxnet as mx
import numpy as np

def small_basic_block(data , num_filter, name_conv):
    conv_1x1 = mx.sym.Convolution(data = data, num_filter = num_filter//4, kernel = (1, 1), 
        stride = (1, 1), pad = (0, 0), name = name_conv+ '/conv_1x1_1', dilate = (1, 1))
    conv_3x1 = mx.sym.Convolution(data = conv_1x1, num_filter = num_filter//4, kernel = (3, 1), 
        stride = (1, 1), pad = (1, 0), name = name_conv+ '/conv_3x1', dilate = (1, 1))
    conv_1x3 = mx.sym.Convolution(data = conv_3x1, num_filter = num_filter//4, kernel = (1, 3), 
        stride = (1, 1), pad = (0, 1), name = name_conv+ '/conv_1x3', dilate = (1, 1))
    conv_1x1 = mx.sym.Convolution(data = conv_1x3, num_filter = num_filter, kernel = (1, 1), 
        stride = (1, 1), pad = (0, 0), name = name_conv+ '/conv_1x1_2', dilate = (1, 1))
    return conv_1x1

def _add_warp_ctc_loss(pred, seq_len, num_label, label):
    """ Adds Symbol.contrib.ctc_loss on top of pred symbol and returns the resulting symbol """
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    return mx.sym.WarpCTC(data=pred, label=label, label_length=num_label, input_length=seq_len)


def _add_mxnet_ctc_loss(pred, seq_len, label):
    """ Adds Symbol.WapCTC on top of pred symbol and returns the resulting symbol """

    arg_shape, out_shapes, aux_shapes = pred.infer_shape(data=(128,3,94,24))
    print(out_shapes)

    pred_ctc = mx.sym.Reshape(data=pred, shape=(-4, seq_len, -1, 0))

    arg_shape, out_shapes, aux_shapes = pred_ctc.infer_shape(data=(128,3,94,24))
    print(out_shapes)

    loss = mx.sym.contrib.ctc_loss(data=pred_ctc, label=label)
    ctc_loss = mx.sym.MakeLoss(loss)

    softmax_class = mx.symbol.SoftmaxActivation(data=pred)
    softmax_loss = mx.sym.MakeLoss(softmax_class)
    softmax_loss = mx.sym.BlockGrad(softmax_loss)
    return mx.sym.Group([softmax_loss, ctc_loss])


def _add_ctc_loss(pred, seq_len, num_label, loss_type):
    """ Adds CTC loss on top of pred symbol and returns the resulting symbol """
    label = mx.sym.Variable('label')
    if loss_type == 'warpctc':
        print("Using WarpCTC Loss")
        sm = _add_warp_ctc_loss(pred, seq_len, num_label, label)
    else:
        print("Using MXNet CTC Loss")
        assert loss_type == 'ctc'
        sm = _add_mxnet_ctc_loss(pred, seq_len, label)
    return sm

def get_symbol(num_classes = 78, loss = 'ctc', seq_len = 24, dtype = 'float32', **kwargs):

    cin = [64, 64, 256]
    cout = [64, 256, 256]

    data = mx.sym.Variable(name = 'data')

    s1_conv1 = mx.sym.Convolution(data = data, num_filter = 64, kernel = (3, 3), stride = (1, 1), 
        pad = (1, 1), name = 'stage1_conv1', dilate = (1, 1))
    s1_bn1 = mx.sym.BatchNorm(data = s1_conv1, name = 'stage1_bn1')
    s1_act1 = mx.sym.Activation(data = s1_bn1, act_type = 'relu', name = 'stage1_relu1')
    s1_pool1 = mx.sym.Pooling(data = s1_act1, global_pool = False, kernel=(3, 3), pool_type = 'max', 
        stride = (1, 1), pad = (1, 1), name = 'stage1_pool1')    
    s1_block1 = small_basic_block(data=s1_pool1, num_filter=64, name_conv='stage1_block1')

    # arg_shape, out_shapes, aux_shapes = s1_block1.infer_shape(**data_shape)
    
    s2_bn1 = mx.sym.BatchNorm(data = s1_block1, name = 'stage2_bn1')
    s2_act1 = mx.sym.Activation(data = s2_bn1, act_type = 'relu', name = 'stage2_relu1')
    s2_pool1 = mx.sym.Pooling(data = s2_act1, global_pool = False, kernel=(3, 3), pool_type = 'max', 
        stride = (2, 1), pad = (1, 1), name = 'stage2_pool1')    
    s2_block1 = small_basic_block(data=s2_pool1, num_filter=256, name_conv='stage2_block1')
    s2_bn2 = mx.sym.BatchNorm(data = s2_block1, name = 'stage2_bn2')
    s2_act2 = mx.sym.Activation(data = s2_bn2, act_type = 'relu', name = 'stage2_relu2')
    s2_block2 = small_basic_block(data = s2_act2, num_filter=256, name_conv='stage2_block2')

    # arg_shape, out_shapes, aux_shapes = s2_block2.infer_shape(**data_shape)

    s3_bn1 = mx.sym.BatchNorm(data = s2_block2, name = 'stage3_bn1')
    s3_act1 = mx.sym.Activation(data = s3_bn1, act_type = 'relu', name = 'stage3_relu1')
    s3_pool1 = mx.sym.Pooling(data = s3_act1, global_pool = False, kernel=(3, 3), pool_type = 'max', 
        stride = (2, 1), pad = (1, 1), name = 'stage3_pool1')
    s3_drop1 = mx.sym.Dropout(data = s3_pool1, p = 0.5)
    s3_conv1 = mx.sym.Convolution(data = s3_drop1, num_filter = 256, kernel = (5, 1), stride = (1, 1), 
        pad = (2, 0), name = 'stage3_conv1', dilate = (1, 1))
    s3_drop2 = mx.sym.Dropout(data = s3_conv1, p = 0.5)
    s3_bn2 = mx.sym.BatchNorm(data = s3_drop2, name = 'stage3_bn2')
    s3_act2 = mx.sym.Activation(data = s3_bn2, act_type = 'relu', name = 'stage3_relu2')
    s3_conv2 = mx.sym.Convolution(data = s3_act2, num_filter = num_classes + 1, kernel = (1, 13), 
        stride = (1, 1), pad = (0, 6), name = 'stage3_conv2', dilate = (1, 1))
    s3_bn3 = mx.sym.BatchNorm(data = s3_conv2, name = 's3_bn3')
    s3_act3 = mx.sym.Activation(data = s3_bn3, act_type = 'relu', name = 'stage3_relu3')

    # data_shape = {'data':(32, 3, 94, 24)}
    # arg_shape, out_shapes, aux_shapes = s3_conv2.infer_shape(**data_shape)
    # arg_shape, out_shapes, aux_shapes = s3_act3.infer_shape(**data_shape)

    # x1 = mx.sym.Pooling(data = data, kernel = (4,1), stride = (4, 1), pool_type = 'avg', 
    #     pad = (1, 0), name = 'pool_x1')
    # x1_bn = mx.sym.BatchNorm(data = x1, name = 'x1_bn')
    # # x1 = mx.sym.square(x1)
    # # cx1 = mx.sym.mean(x1)
    # # x1 = mx.sym.broadcast_div(x1, cx1)


    # x2 = mx.sym.Pooling(data = s1_block1, kernel = (4,1), stride = (4, 1), pool_type = 'avg', 
    #     pad = (1, 0), name = 'pool_x2')
    # x2_bn = mx.sym.BatchNorm(data = x2, name = 'x2_bn')
    # # x2 = mx.sym.square(x2)
    # # cx2 = mx.sym.mean(x2)
    # # x2 = mx.sym.broadcast_div(x2, cx2)

    # x3 = mx.sym.Pooling(data = s2_block2, kernel = (2,1), stride = (2, 1),  pool_type = 'avg', 
    #     pad = (1, 0), name = 'pool_x3')
    # x3_bn = mx.sym.BatchNorm(data = x3, name = 'x3_bn')
    # # x3 = mx.sym.square(x3)
    # # cx3 = mx.sym.mean(x3)
    # # x3 = mx.sym.broadcast_div(x3, cx3)
    # #x3 = mx.sym.L2Normalization(x3)

    # x4_bn = mx.sym.BatchNorm(data = s3_act3, name = 'x4_bn')
    # # x4 = mx.sym.square(s3_act3)
    # # cx4 = mx.sym.mean(x4)
    # # x4 = mx.sym.broadcast_div(x4, cx4)
    # # arg_shape, out_shapes, aux_shapes = x4.infer_shape(**data_shape)
    # #x4 = mx.sym.L2Normalization(s3_act3)

    # x_all = mx.sym.concat(x1, x2, x3, x4, dim = 1)
    # x_all = mx.sym.concat(x1_bn, x2_bn, x3_bn, x4_bn, dim = 1)
    # x_all = s3_act3

    # conv_last = mx.sym.Convolution(data = x_all, num_filter = num_classes + 1, kernel = (1, 1), 
    #     stride = (1, 1), pad = (0,0), name = 'conv_last', dilate = (1, 1))
    conv_last = mx.sym.Convolution(data = s3_act3, num_filter = num_classes + 1, kernel = (1, 1), 
        stride = (1, 1), pad = (0,0), name = 'conv_last', dilate = (1, 1))
    # logits_b_c_s = mx.sym.mean(conv_last, axis=3)
    logits_b_c_s = mx.sym.Pooling(data = conv_last, kernel = (1,24), stride = (1, 1),  pool_type = 'avg', 
        pad = (0, 0), name = 'pool_last')

    arg_shape, out_shapes, aux_shapes = logits_b_c_s.infer_shape(data=(128,3,94,24))
    print('logits_b_c_s:',out_shapes)

    logits_s_b_c = mx.sym.transpose(data=logits_b_c_s,axes=(2,0,1,3))
    # logits_s_b_c = mx.sym.transpose(data=logits_b_c_s,axes=(2,0,1))
    arg_shape, out_shapes, aux_shapes = logits_s_b_c.infer_shape(data=(128,3,94,24))
    print('logits_b_s_c:',out_shapes)

    logits = mx.sym.reshape(data=logits_s_b_c, shape=(-3, 0))
    # arg_shape, out_shapes, aux_shapes = logits.infer_shape(**data_shape)

    #mx.viz.plot_network(logits, shape={"data" : (1, 3, 94, 24)}, node_attrs={"shape":'rect',"fixedsize":'false'}).view('lprnet')

    if loss:
        # Training mode, add loss
        return _add_ctc_loss(logits, seq_len, 79, loss)
        # ctc_loss = _add_ctc_loss(logits, seq_len, num_classes + 1, loss)
        # data_shape = {'data':(1, 3, 94, 24)}
        # arg_shape, out_shapes, aux_shapes = ctc_loss.infer_shape(**data_shape)
        # return ctc_loss
    else:
        # Inference mode, add softmax
        return mx.sym.softmax(data=logits, name='softmax')

    # return logits
    

if __name__ == '__main__':
    net = get_symbol(num_classes = 78)
