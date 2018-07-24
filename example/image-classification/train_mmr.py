# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train vehicle make model recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 4)
    parser.set_defaults(
        # network
        network          = 'peleenet',
        num_layers       = 18, # resnet specific option
        num_classes      = 3341,
        multiplier       = 1.0, # mobilenet-v2 specific option
        visibility       = 0, 
        summarywriter    = 0, # log accuracy, cross-entropy and weights with SummaryWriter 
        # data
        num_examples     = 294533,
        image_shape      = '3,224,224',
        # image_shape      = '3,299,299', # inception-v3 specific
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        data_train       = '/opt/data/mmr/train_head.rec',
        # data_train       = '/opt/data/mmr/train_356.rec', # inception-v3 specific
        # rgb_mean       = '64.0861,62.3341,72.3381',
        mean_img         = '/opt/data/mmr/mean_head.bin',
        mean_img_dir     = '/opt/data/mmr/mean_head',
        # mean_img         = '/opt/data/mmr/mean_299.bin', # inception-v3 specific
        convert_numpy    = 1, # switch for whether convet mean.bin to mean.npy, set 0 to close convert, 1 to open convert
        # train
        batch_size       = 128,
        disp_batches     = 200,
        num_epochs       = 70,
        # load_epoch       = 9,
        optimizer        = 'sgd',
        loss             = 'ce',
        lr               = 0.1,
        lr_factor        = 0.1,
        lr_step_epochs   = '20, 40, 60',
        mom              = 0.9,
        wd               = 0.00004,
        dtype            = 'float32',
        model_prefix     = '/opt/models/mxnet/mmr/mmr_peleenet/mmr_peleenet',
        gpus             = '0, 1',
        flush_secs       = 180
        # additional parameters for large batch sgd
        # macrobatch_size  = 96,
        # warmup_strategy  = 'linear'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args)) # not mobilenet-v2
    # sym = net.get_symbol(args.num_classes, args.multiplier) # mobilenet-v2 specific 

    # model structure visualization
    if args.visibility == 1:
        mx.viz.plot_network(sym, shape={"data":(1, 3, 224, 224)}, node_attrs={"shape":'rect',"fixedsize":'false'}).render(args.network)
        exit()

    # train
    fit.fit(args, sym, data.get_rec_iter)
