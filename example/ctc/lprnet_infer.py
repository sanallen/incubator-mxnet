# -*- coding: utf-8 -*-
from ctc_metrics import CtcMetrics
import cv2
from hyperparams import Hyperparams
import lstm
import mxnet as mx
import numpy as np
from train_lprnet import LPRIter

CHARS = [
         u'_', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
         u'9', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'J', 
         u'K', u'L', u'M', u'N', u'P', u'Q', u'R', u'S', u'T', u'U', 
         u'V', u'W', u'X', u'Y', u'Z',
         u'京', u'沪', u'津', u'渝', u'冀', u'晋', u'蒙', u'辽', u'吉', u'黑',
         u'苏', u'浙', u'皖', u'闽', u'赣', u'鲁', u'豫', u'鄂', u'湘', u'粤',
         u'桂', u'琼', u'川', u'贵', u'云', u'藏', u'陕', u'甘', u'青', u'宁',
         u'新', u'港', u'澳', u'台', u'警', u'学', u'使', u'领', u'电', u'通', 
         u'边', u'挂', u'民', u'航',
         ]

# CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

# NUM_CHARS = len(CHARS)

def load_module(prefix, epoch, data_shapes):
    """
    Loads the model from checkpoint specified by prefix and epoch, binds it
    to an executor, and sets its parameters and returns a mx.mod.Module
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # We don't need CTC loss for prediction, just a simple softmax will suffice.
    # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
    pred_fc = sym.get_internals()['softmaxactivation0_output']
    #arg_shape, out_shapes, aux_shapes = pred_fc.infer_shape(data=(128,3,94,24))
    #print(out_shapes)

    #sym = mx.sym.softmax(data=pred_fc)

    mod = mx.mod.Module(symbol=pred_fc, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=False)
    return mod

def index_2_char(seq):
    # pass
    # if not isinstance(seq[0], int):
    #     seq = [int(item) for item in seq]
    char_list = []
    for item in seq:
        char_list.append(CHARS[int(item)])
    string = str("".join(char_list))+ "("+str(len(char_list))+ ") "
    return string
    

def main():
    batch_size = 64
    seq_len = 24

    test = LPRIter(
        path_imgrec         = '/opt/data/plate/rec/test.rec',
        path_imglist        = '/opt/data/plate/rec/resized_plate_test.txt',
        label_width         = 8,
        mean_img            = '/opt/data/plate/rec/mean_plate.bin',
        data_shape          = (3,94,24),
        batch_size          = batch_size,
        augment             = False
    )

    mod = load_module('/opt/models/mxnet/plate/plate_lprnet/lprnet_sgd_drop_fc_augment_wd.0005', 500, test.provide_data)
    # mod = load_module('/opt/incubator-mxnet/example/ctc/model/plate_rec', 1981, test.provide_data)
    # mod = load_module('/opt/incubator-mxnet/example/ctc/model/lprnet_sgd', 500, test.provide_data)
    batchnum = 0
    hit = 0.
    total = 0.

    test_error_list = []

    while True:
        try:
            batch = test.next()
            batchnum += 1
            mod.forward(batch)
            preds = mod.get_outputs()[0].asnumpy()
            preds = np.argmax(preds, axis=-1).reshape((seq_len,batch_size)).transpose((1,0))
            
            for i in range(batch_size):
                p = CtcMetrics.ctc_label(preds[i].tolist())
                #label_pred = [p - 1 for p in label_pred]
                #print(label_pred)
                l = CtcMetrics._remove_blank(batch.label[0].asnumpy()[i])
                is_match = False
                if len(p) == len(l):
                    match = True
                    for k, _ in enumerate(p):
                        if p[k] != int(l[k]):
                            match = False
                            break
                    if match:
                        hit += 1.0
                        is_match = True
                if not is_match:
                    decode_l = index_2_char(l)
                    decode_p = index_2_char(p)
                    print_str = decode_l + '\t'+ decode_p
                    print("label:%s\tpred:%s" % (decode_l, decode_p))
                    # print("pred:%s" % decode_p)
                    test_error_list.append(print_str)
                total += 1.0

        except StopIteration:
            break
    ferror = open('/opt/incubator-mxnet/example/ctc/test_error_list.txt', 'w')
    for file_str in test_error_list:
        ferror.writelines(file_str+'\n')
    ferror.close()
    print(hit/total, hit, total)

if __name__ == '__main__':
    main()
