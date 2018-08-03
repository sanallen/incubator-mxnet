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

# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import importlib
import mxnet as mx
from dataset.iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric
import logging
from symbol.symbol_factory import get_symbol
import find_wrong_detection
from collections import Counter
from matplotlib import pyplot as plt
import datetime

def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin = 0, Ymax = 0, ovp_thresh = 0.5, 
    netname = 'legacy_pelee_SSD_v2x'):
    data = plt.hist(myList, 100)
    Ymax = int(max(data[0])*1.1)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    log = datetime.datetime.now().strftime('%Y-%m-%d')
    plt.savefig('./model/iou_distribution/%s_%s_ovp_thresh_%.2f.jpg' % (netname, log, ovp_thresh))
    plt.close()

def evaluate_net(net, path_imgrec, num_classes, mean_pixels, data_shape,
                 model_prefix, epoch, path_img, ctx=mx.cpu(), batch_size=1,
                 path_imglist="", nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False, class_names=None,
                 voc07_metric=False):
    """
    evalute network given validation record file

    Parameters:
    ----------
    net : str or None
        Network name or use None to load from json without modifying
    path_imgrec : str
        path to the record validation file
    path_imglist : str
        path to the list file to replace labels in record file, optional
    num_classes : int
        number of classes, not including background
    mean_pixels : tuple
        (mean_r, mean_g, mean_b)
    data_shape : tuple or int
        (3, height, width) or height/width
    model_prefix : str
        model prefix of saved checkpoint
    epoch : int
        load model epoch
    ctx : mx.ctx
        mx.gpu() or mx.cpu()
    batch_size : int
        validation batch size
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : boolean
        whether suppress different class objects
    ovp_thresh : float
        AP overlap threshold for true/false postives
    use_difficult : boolean
        whether to use difficult objects in evaluation if applicable
    class_names : comma separated str
        class names in string, must correspond to num_classes if set
    voc07_metric : boolean
        whether to use 11-point evluation as in VOC07 competition
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    model_prefix += '_' + str(data_shape[1])

    netname = net

    # iterator
    eval_iter = DetRecordIter(path_imgrec, batch_size, data_shape, mean_pixels=mean_pixels,
                              path_imglist=path_imglist, **cfg.valid)
    # model params
    load_net, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # network
    if net is None:
        net = load_net
    else:
        net = get_symbol(net, data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    mod.set_params(args, auxs, allow_missing=False, force_init=True)

    # # run evaluation
    # if voc07_metric:
    #     metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
    # else:
    #     metric = MApMetric(ovp_thresh, use_difficult, class_names)
    # results = mod.score(eval_iter, metric, num_batch=None)
    # for k, v in results:
    #     print("{}: {}".format(k, v))    

    predict_results = mod.predict(eval_iter, merge_batches = True)
    preds = predict_results[0]
    labels = predict_results[1]

    (flags, ious) = find_wrong_detection.find_wrong_detection(labels, preds, path_imglist, path_img, ovp_thresh = ovp_thresh)
    flags_dict = {0:'correct', 1:'lower iou', 2:'wrong class'}
    flag_count = Counter(flags)
    for flag in set(flags):
        print ("%s image number is : %d"%(flags_dict[flag], flag_count[flag]))
    if not os.path.exists('./model/iou_distribution'):
        os.mkdir('./model/iou_distribution')
    xmin = min(ious) - 0.1 if min(ious) > 0.1 else 0
    xmax = max(ious) + 0.1 if min(ious) < 0.9 else 1
    title = "iou distribution" + '(ovp_thresh = %.2f)' % (ovp_thresh)
    draw_hist(ious, title, "iou", "image number", xmin, xmax, 0, len(ious)/20, ovp_thresh, netname)
   