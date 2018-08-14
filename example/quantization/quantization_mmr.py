# -*- coding: utf-8 -*-

import argparse
import os
import logging
from common import modelzoo
import mxnet as mx
import time
from mxnet.contrib.quantization import *
from mxboard import SummaryWriter
import shutil


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--model', dest='model', type=str, default ='mmr_peleenet', 
                        help='model name')
    parser.add_argument('--model-path', dest='model_path', type=str, default ='/opt/incubator-mxnet/example/quantization/model/', 
                        help='where the model is')
    parser.add_argument('--epoch', dest='epoch', type=int, default =70, 
                        help='the epoch of trained F32 model')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--calib-dataset', type=str, default='/opt/incubator-mxnet/example/quantization/data/calib_mmr.rec',
                        help='path of the calibration dataset')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60,
                        help='number of threads for data decoding')
    parser.add_argument('--num-calib-batches', type=int, default=100,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=True,
                        help='excluding quantizing the first conv layer since the'
                             ' number of channels is usually not a multiple of 4 in that layer'
                             ' which does not satisfy the requirement of cuDNN')
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
    parser.add_argument('--calib-mode', type=str, default='entropy',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='int8', 
                        choices=['int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--visibility', dest = 'visibility', action='store_true', default=False,
                        help='plot the structure of the network')
    parser.add_argument('--log', dest='log_file', type=str, default='/opt/incubator-mxnet/example/quantization/log/mmr_peleenet_quantize-'+time.strftime("%y-%m-%d")+'.log',
                        help='save quantizing log to file')
    parser.add_argument('--mean-img', dest='mean_img', type=str, 
                        default='/opt/data/mmr/mean_head.bin', help='mean image to subtract')
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        os.mknod(args.log_file)
    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        logger.addHandler(fh)

    shutil.rmtree('/opt/incubator-mxnet/logs') # clear the previous logs
    os.mkdir('/opt/incubator-mxnet/logs')
    sw = SummaryWriter(logdir='/opt/incubator-mxnet/logs', flush_secs = 180)

    logger.info('shuffle_dataset=%s' % args.shuffle_dataset)

    calib_mode = args.calib_mode
    logger.info('calibration mode set to %s' % calib_mode)

    model_prefix = os.path.join(args.model_path, args.model)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.epoch)

    if args.visibility == 1: 
        mx.viz.plot_network(sym, shape={"data":(1, 3, 224, 224)}, 
                            node_attrs={"shape":'rect',"fixedsize":'false'}).render(args.model)
        exit()

    # get batch size
    batch_size = args.batch_size
    logger.info('batch size = %d for calibration' % batch_size)

    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    if calib_mode != 'none':
        logger.info('number of batches = %d for calibration' % num_calib_batches)

    # get number of threads for decoding the dataset
    data_nthreads = args.data_nthreads

    # get image shape
    image_shape = args.image_shape

    exclude_first_conv = args.exclude_first_conv
    excluded_sym_names = []

    if args.model == 'mmr_peleenet':
        # gpu 与cpu 实现的量化操作不同，见ppt4
        if args.ctx == 'gpu':
            calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                                     or name.find('classifier') != -1)
        else:
            calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1)
            excluded_sym_names += ['classifier_0', 'classifier_1']
        if exclude_first_conv:
            excluded_sym_names += ['stem1/conv']
    else:
        raise ValueError('model %s is not supported in this script' % args.model)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    if calib_mode == 'none':
        logger.info('Quantizing FP32 model %s' % args.model)
        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                       ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                       calib_mode=calib_mode, quantized_dtype=args.quantized_dtype,
                                                       logger=logger)
        sym_name = '%s-symbol.json' % (prefix + '-quantized')
        save_symbol(sym_name, qsym, logger)
    else:
        logger.info('Creating ImageRecordIter for reading calibration dataset')
        data = mx.io.ImageRecordIter(path_imgrec=args.calib_dataset,
                                     label_width=1,
                                     preprocess_threads=data_nthreads,
                                     batch_size=batch_size,
                                     data_shape=data_shape,
                                     label_name=label_name,
                                     rand_crop=False,
                                     rand_mirror=False,
                                     shuffle=args.shuffle_dataset,
                                     shuffle_chunk_seed=args.shuffle_chunk_seed,
                                     seed=args.shuffle_seed,
                                     mean_img=args.mean_img)

        cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                        ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                        calib_mode=calib_mode, calib_data=data,
                                                        num_calib_examples=num_calib_batches * batch_size,
                                                        calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                        logger=logger)
        if calib_mode == 'entropy':
            suffix = '-quantized-%dbatches-entropy' % num_calib_batches
        elif calib_mode == 'naive':
            suffix = '-quantized-%dbatches-naive' % num_calib_batches
        else:
            raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                             % calib_mode)
        sym_name = '%s-symbol.json' % (model_prefix + suffix)
        save_symbol(sym_name, cqsym, logger)

    sw.add_graph(cqsym)

    param_name = '%s-%04d.params' % (model_prefix + '-quantized', args.epoch)
    save_params(param_name, qarg_params, aux_params, logger)
    sw.close()
