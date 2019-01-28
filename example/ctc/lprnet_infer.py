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

def load_module(prefix, epoch, data_shapes):
    """
    Loads the model from checkpoint specified by prefix and epoch, binds it
    to an executor, and sets its parameters and returns a mx.mod.Module
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # We don't need CTC loss for prediction, just a simple softmax will suffice.
    # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
    
    # pred_fc = sym.get_internals()['softmaxactivation0_output']
    pred_fc = sym.get_internals()['softmax_output']

    mod = mx.mod.Module(symbol=pred_fc, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=False)
    return mod

def index_2_char(seq):
    char_list = []
    for item in seq:
        char_list.append(CHARS[int(item)])
    string = str("".join(char_list))+ "("+str(len(char_list))+ ") "
    return string

def vec_2_str(vec):
    str_vec = []
    for item in vec:
        str_vec.append(str(item))
    rst_str = "\t".join(str_vec) + "\n"
    return rst_str


def main():
    batch_size = 1
    seq_len = 24

    test = LPRIter(
        path_imgrec         = '/mnt/ExtraSSD/data/plate/color_rec/vpr_blue_head_pad.rec',
        path_imglist        = '/mnt/ExtraSSD/data/plate/color_rec/vpr_blue_head_pad.txt',
        label_width         = 8,
        # mean_img            = '/opt/data/plate/rec/mean_plate.bin',
        data_shape          = (3,94,24),
        batch_size          = batch_size,
        augment             = False,
        shuffle             = False
    )

    mod = load_module('/opt/models/mxnet/plate/plate_lprnet/deploy_lprnet_concat', 500, test.provide_data)

    batchnum = 0
    hit = 0.
    total = 0.

    test_error_list = []

    # fsequence = open('/opt/incubator-mxnet/example/ctc/sequence.txt', 'w')

    while True:
        try:
            batch = test.next()
            input_data = batch.data[0].asnumpy()
            batchnum += 1
            mod.forward(batch)
            output = mod.get_outputs()[0].asnumpy()
            preds = np.argmax(output, axis=-1).reshape((seq_len,batch_size)).transpose((1,0))
            # greedy_str = vec_2_str(preds[0])
            # fsequence.writelines("greedy sequence:\n")
            # fsequence.writelines(greedy_str)
            for i in range(batch_size):
                p = CtcMetrics.ctc_label(preds[i].tolist())
                # best_str = vec_2_str(p)
                # fsequence.writelines("best sequence:\n")
                # fsequence.writelines(best_str)
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
                    test_error_list.append(print_str)
                total += 1.0

        except StopIteration:
            break
    ferror = open('/opt/incubator-mxnet/example/ctc/test_error_list.txt', 'w')
    for file_str in test_error_list:
        ferror.writelines(file_str+'\n')
    ferror.close()
    print(hit/total, hit, total)
    # fsequence.close()

if __name__ == '__main__':
    main()
