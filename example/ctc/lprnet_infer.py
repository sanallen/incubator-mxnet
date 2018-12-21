from ctc_metrics import CtcMetrics
import cv2
from hyperparams import Hyperparams
import lstm
import mxnet as mx
import numpy as np
from train_lprnet import LPRIter

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

def main():
    batch_size = 64
    seq_len = 24

    test = LPRIter(
        path_imgrec         = '/opt/data/plate/test.rec',
        path_imglist        = '/opt/data/plate/resized_plate_test.txt',
        label_width         = 8,
        mean_img            = '/opt/data/plate/mean_plate.bin',
        data_shape          = (3,94,24),
        batch_size          = batch_size,
    )

    mod = load_module('plate3_rec', 367, test.provide_data)
    batchnum = 0
    hit = 0.
    total = 0.

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
                    print(l,p)
                total += 1.0
        except StopIteration:
            break
    print(hit/total, hit, total)
''' preds = mod.predict(eval_data=test,num_batch=1).asnumpy()

    preds = np.argmax(preds, axis=-1).reshape((seq_len,batch_size)).transpose((1,0))
    print(preds.shape)



    for i in range(batch_size):
        label_pred = CtcMetrics.ctc_label(preds[i].tolist())
        label_pred = [p - 1 for p in label_pred]
        print(label_pred)
'''
if __name__ == '__main__':
    main()
