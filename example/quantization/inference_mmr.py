import argparse
import logging
import os
import time
import mxnet as mx
from mxnet import nd
from mxnet.contrib.quantization import *


def load_model(symbol_file, param_file, logger=None):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if logger is not None:
        logger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params


def advance_data_iter(data_iter, n):
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False


def score(sym, arg_params, aux_params, data, devs, label_name, max_num_examples, logger=None):
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k=5)]
    if not isinstance(metrics, list):
        metrics = [metrics, ]
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break
    speed = num / (time.time() - tic)

    if logger is not None:
        logger.info('Finished inference with %d images' % num)
        logger.info('Finished with %f images per second', speed)
        for m in metrics:
            logger.info(m.get())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--symbol-file', type=str, help='symbol file path',
                        default='/opt/incubator-mxnet/example/quantization/model/mmr_peleenet-quantized-100batches-entropy-symbol.json')
    parser.add_argument('--param-file', type=str, help='param file path',
                        default='/opt/incubator-mxnet/example/quantization/model/mmr_peleenet-quantized-0070.params')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--dataset', type=str, default='/opt/incubator-mxnet/example/quantization/data/test_calib_mmr.rec', help='dataset path')
    parser.add_argument('--mean-img', dest='mean_img', type=str, 
                        default='/opt/data/mmr_new/mean_head.bin', help='mean image to subtract')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, default=500, help='number of images used for inference')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')

    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)
    
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    # creating data iterator
    data = mx.io.ImageRecordIter(path_imgrec=args.dataset,
                                 label_width=1,
                                 preprocess_threads=data_nthreads,
                                 batch_size=batch_size,
                                 data_shape=data_shape,
                                 label_name=label_name,
                                 rand_crop=False,
                                 rand_mirror=False,
                                 shuffle=True,
                                 shuffle_chunk_seed=3982304,
                                 seed=48564309,
                                 mean_img=args.mean_img)

    # loading model
    sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)

    # make sure that fp32 inference works on the same images as calibrated quantized model
    logger.info('Skipping the first %d batches' % args.num_skipped_batches)
    data = advance_data_iter(data, args.num_skipped_batches)

    num_inference_images = args.num_inference_batches * batch_size
    logger.info('Running model %s for inference' % symbol_file)
    score(sym, arg_params, aux_params, data, [ctx], label_name,
          max_num_examples=num_inference_images, logger=logger)
