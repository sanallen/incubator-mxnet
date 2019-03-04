#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import argparse
import mxnet as mx
import os
import time
from train.train_net import train_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

    # parser.add_argument('--train-path', dest='train_path', help='train record to use',
    #                     default=os.path.join(os.getcwd(), 'data', 'train.rec'), type=str)
    parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default='/mnt/ExtraSSD/data/mxnet_ssd/train.rec', type=str)
                        # default='/mnt/ExtraSSD/data/mxnet_ssd/train_head.rec', type=str)
    parser.add_argument('--train-list', dest='train_list', help='train list to use',
                        default="", type=str)
    # parser.add_argument('--val-path', dest='val_path', help='validation record to use',
    #                     default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default='/mnt/ExtraSSD/data/mxnet_ssd/test.rec', type=str)
                        # default='/mnt/ExtraSSD/data/mxnet_ssd/val_head.rec', type=str)
    parser.add_argument('--val-list', dest='val_list', help='validation list to use',
                        default="", type=str)
    # parser.add_argument('--network', dest='network', type=str, default='mobilenetv2',
    # parser.add_argument('--network', dest='network', type=str, default='legacy_plate_pelee_head',
    #                     help='which network to use')
    parser.add_argument('--network', dest='network', type=str, default='legacy_plate_pelee',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default='', type=str)
    # parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
    #                     default='/opt/incubator-mxnet/example/ssd/model/ssd_legacy_plate_pelee_head_640', type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default='/opt/incubator-mxnet/example/ssd/model/ssd', type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=280, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging', 
                        default=50, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=480,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=350,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-img', dest='mean_img', type=str, 
                        # default='/mnt/ExtraSSD/data/detection/mean_head.bin', help='mean image to subtract')
                        default='/mnt/ExtraSSD/data/detection/mean.bin', help='mean image to subtract')
    parser.add_argument('--mean-img-dir', dest='mean_img_dir', type=str, 
                        # default='/mnt/ExtraSSD/data/detection/mean_head', help='mean image in numpy')
                        default='/mnt/ExtraSSD/data/detection/mean', help='mean image in numpy')
    parser.add_argument('--convert_numpy', dest='convert_numpy', type=int, 
                        default=1, help='mean image in numpy')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, 
                        default='20,30,40,45,50,55,60,65,70,75,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=float, default=0.8,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, 
                        default='/opt/incubator-mxnet/example/ssd/model/peleenet/'+ 'train-'+time.strftime("%y-%m-%d")+'.log',
                        help='save training log to file')
    # parser.add_argument('--log', dest='log_file', type=str, default=os.path.join(os.getcwd(), 'model', 'legacy_pelee', 'train-'+time.strftime("%y-%m-%d")+'.log'),
    #                     help='save training log to file')                        
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--num-class', dest='num_class', type=int, default=1,
                        help='number of classes')
    parser.add_argument('--num-example', dest='num_example', type=int, default=34977,
                        help='number of image examples')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='LPRrect',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.8,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', action='store_true',  default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', action='store_true', default=True,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--no-voc07', dest='use_voc07_metric', action='store_false',
                        help='dont use PASCAL VOC 07 11-point metric')
    parser.add_argument('--summarywriter', dest='summarywriter', default=False,
                        help='1 means log weight and evaluation metric with summarywriter')
    parser.add_argument('--flush_secs', dest='flush_secs', type=int, default=360,
                        help='How often, in seconds, to flush the pending events and summaries to disk.')
    args = parser.parse_args()
    return args

def parse_class_names(args):
    """ parse # classes and class_names if applicable """
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
            # try to open it to read class names
            with open(args.class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # class names if applicable
    class_names = parse_class_names(args)
    # start training
    if not os.path.exists(os.path.dirname(args.log_file)):
        os.makedirs(os.path.dirname(args.log_file))

    train_net(args.network, args.train_path,
              args.num_class, args.batch_size,
              args.data_shape, args.mean_img, args.mean_img_dir,
              args.resume, args.finetune, args.pretrained,
              args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              args.frequent, args.learning_rate, args.momentum, args.weight_decay,
              args.lr_refactor_step, args.lr_refactor_ratio,
              convert_numpy = args.convert_numpy,
              val_path=args.val_path,
              num_example=args.num_example,
              class_names=class_names,
              label_pad_width=args.label_width,
              freeze_layer_pattern=args.freeze_pattern,
              iter_monitor=args.monitor,
              monitor_pattern=args.monitor_pattern,
              log_file=args.log_file,
              nms_thresh=args.nms_thresh,
              force_nms=args.force_nms,
              ovp_thresh=args.overlap_thresh,
              use_difficult=args.use_difficult,
              voc07_metric=args.use_voc07_metric,
              summarywriter=args.summarywriter)
