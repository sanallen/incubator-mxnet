# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
import importlib
# from dataset.iterator import DetRecordIter
from config.config import cfg
import logging
from symbol.symbol_factory import get_symbol
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import cv2
# from evaluate.evaluate_net import evaluate_net

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--list-path', dest='list_path', help='which list file to use',
                        default='/mnt/ExtraSSD/data/mxnet_ssd/blue.lst', type=str)
                        # default='/mnt/ExtraSSD/data/mxnet_ssd/blue_head.lst', type=str)
    parser.add_argument('--img-path', dest='img_path', help='where the image is',
                        default='/mnt/ExtraSSD/data/plate/vpr_blue_voc', type=str)
                        # default='/mnt/ExtraSSD/data/car_head_test', type=str)                   
    parser.add_argument('--network', dest='network', type=str, default='legacy_plate_pelee',
                        help='which network to use')
    parser.add_argument('--num-class', dest='num_class', type=int, default=1,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='LPRrect',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=235, type=int)
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default='/opt/incubator-mxnet/example/ssd/model/ssd_', type=str)
    parser.add_argument('--gpus', dest='gpu_id', help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--cpu', dest='cpu', help='use cpu to evaluate, this can be slow',
                        action='store_true')
    # parser.add_argument('--data-shape', dest='data_shape', type=int, default=320,
    # parser.add_argument('--data-shape', dest='data_shape', type=int, default=640,
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=480,
                        help='set image shape')
    parser.add_argument('--mean-img', dest='mean_img', type=str, 
                        # default='/mnt/ExtraSSD/data/detection/mean_head.bin', help='mean image to subtract')
                        # default='/mnt/ExtraSSD/data/detection/mean.bin', help='mean image to subtract')
                        default='', help='mean image to subtract')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # choose ctx
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]

    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network
    else:
        prefix = args.prefix

    # args
    if isinstance(args.data_shape, int):
        data_shape = (3, args.data_shape, args.data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    model_prefix = prefix + '_' + str(data_shape[1])

    fmiss = open('/opt/incubator-mxnet/example/ctc/miss_detection_list.txt', 'w')
    # for file_str in test_error_list:
    #     ferror.writelines(file_str+'\n')
    # ferror.close()
    # print(hit/total, hit, total)


    # iterator
    list_path = args.list_path
    img_path = args.img_path
    network = args.network
    num_classes = args.num_class

    # model params
    load_net, model_args, model_auxs = mx.model.load_checkpoint(model_prefix, args.epoch)

    net = get_symbol(network, data_shape[1], num_classes=num_classes,
        nms_thresh=0.45, force_suppress=False)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    mod = mx.mod.Module(net, label_names=('label',), context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(for_training=False,data_shapes=[('data', (1,3,data_shape[1],data_shape[1]))], label_shapes=[('label', (1,1,6))])
    mod.set_params(model_args, model_auxs, allow_missing=False, force_init=True)

    # head_img_path = os.path.join(img_path, 'head')
    head_img_path = '/mnt/ExtraSSD/data/plate/plate_box'
    if os.path.exists(head_img_path):
        shutil.rmtree(head_img_path)
    os.mkdir(head_img_path)

    fp = open(list_path)
    listlines = fp.readlines()

    cnt = 0
    for lines in listlines:
        imgname = lines.split('\t')[-1]
        # 去除换行符
        imgname = imgname.replace('\r','').replace('\n','').replace('\t','')
        # img_name_list.append(imgname)
        
        try:
            # img = cv2.imread(os.path.join(img_path, imgname))
            # resized_img = cv2.resize(img, (data_shape[1], data_shape[1]))
            # trans_img = resized_img.transpose(2, 0, 1)

            img = Image.open(os.path.join(img_path, imgname))
            img_480 = img.resize((data_shape[1], data_shape[1]), Image.ANTIALIAS)   
            np_img = np.asarray(img_480, dtype=np.float32)
            trans_img = np_img.transpose(2, 0, 1)
            
            val_img = trans_img.reshape(1, 3, data_shape[1], data_shape[1])
            val = mx.io.NDArrayIter(data=val_img)

            predict_results = mod.predict(val)[0]
            # print(predict_results.shape)
            max_prob_box = predict_results[0][0]
            # print(max_prob_box)

            np_original_img = np.asarray(img, dtype=np.uint8)
            hei, wid, cha= np_original_img.shape
            # hei, wid, cha= img.shape

            xmin = int((max_prob_box[2]*wid).asnumpy())
            ymin = int((max_prob_box[3]*hei).asnumpy())
            xmax = int((max_prob_box[4]*wid).asnumpy())
            ymax = int((max_prob_box[5]*hei).asnumpy())
            ### 外扩
            box_w = xmax-xmin
            box_h = ymax-ymin
            # ratio = 0.05
            ratio = 0.02
            ext_x = int(box_w*ratio)
            ext_y = int(box_h*ratio)
            xmin = max([xmin-ext_x, 0])
            xmax = min([xmax+ext_x, wid-1])
            ymin = max([ymin-ext_y, 0])
            ymax = min([ymax+ext_y, hei-1])

            np_head = np_original_img[ymin:ymax, xmin:xmax]
            # np_head = img[ymin:ymax, xmin:xmax]
            # cv2.imwrite(os.path.join(head_img_path, imgname.split('/')[-1]), np_head)

            img_head = Image.fromarray(np_head.astype('uint8'))
            img_head.save(os.path.join(head_img_path, imgname.split('/')[-1]))
        except Exception as e:
            print(e)
            print(imgname)
            print(max_prob_box)
            fmiss.writelines(imgname+'\n')
            print(cnt)   
        if cnt%100 ==0:        
            print(cnt)
        cnt += 1
    fmiss.close()
   