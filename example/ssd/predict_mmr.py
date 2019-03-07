# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
import importlib
from config.config import cfg
import logging
from symbol.symbol_factory import get_symbol
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--list-path', dest='list_path', help='which list file to use',
                        default='/mnt/ExtraSSD/data/mxnet_ssd/train&test.lst', type=str)
                        # default='/mnt/ExtraSSD/data/车头错检/error.lst', type=str)                  
                        # default='/opt/nfs/data/test_img.lst', type=str)
    parser.add_argument('--img-path', dest='img_path', help='where the image is',
                        default='/mnt/ExtraSSD/data/VOC-LPR/', type=str)
                        # default='/mnt/ExtraSSD/data/车头错检/', type=str)
                        # default='/opt/nfs/data/', type=str)                    
    parser.add_argument('--network', dest='network', type=str, default='legacy_pelee_SSD_v1',
                        help='which network to use')
    parser.add_argument('--num-class', dest='num_class', type=int, default=8,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='LPRrect',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=249, type=int)
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default='/opt/incubator-mxnet/example/ssd/model/ssd_', type=str)
    parser.add_argument('--gpus', dest='gpu_id', help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--cpu', dest='cpu', help='use cpu to evaluate, this can be slow',
                        action='store_true')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=320,
    # parser.add_argument('--data-shape', dest='data_shape', type=int, default=640,
                        help='set image shape')
    parser.add_argument('--mean-img', dest='mean_img', type=str, 
                        default='/mnt/ExtraSSD/data/detection/mean_head.bin', help='mean image to subtract')

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

    fmiss = open('/opt/incubator-mxnet/example/ctc/miss_detection_car_list_train.txt', 'w')
    fhead = open('/opt/incubator-mxnet/example/ctc/car_head_train.txt', 'w')

    # iterator
    list_path = args.list_path
    img_path = args.img_path
    network = args.network
    num_classes = args.num_class

    # model params
    load_net, model_args, model_auxs = mx.model.load_checkpoint(model_prefix, args.epoch)

    net = get_symbol(network, data_shape[1], num_classes=num_classes,
        nms_thresh=0.45, force_suppress=True)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    mod = mx.mod.Module(net, label_names=('label',), context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(for_training=False,data_shapes=[('data', (1,3,data_shape[1],data_shape[1]))], label_shapes=[('label', (1,1,6))])
    mod.set_params(model_args, model_auxs, allow_missing=False, force_init=True)

    draw_head_path = '/mnt/ExtraSSD/data/draw_head_train'
    head_img_path = '/mnt/ExtraSSD/data/car_head_train'
    if os.path.exists(head_img_path):
        shutil.rmtree(head_img_path)
    os.mkdir(head_img_path)
    if os.path.exists(draw_head_path):
        shutil.rmtree(draw_head_path)
    os.mkdir(draw_head_path)

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
            trans_img[0,:,:] = trans_img[0,:,:] - 123
            trans_img[1,:,:] = trans_img[1,:,:] - 117
            trans_img[2,:,:] = trans_img[2,:,:] - 104

            # tmp = trans_img[0,:,:]
            # trans_img[2,:,:] = trans_img[0,:,:]
            # trans_img[0,:,:] = tmp
            
            val_img = trans_img.reshape(1, 3, data_shape[1], data_shape[1])
            val = mx.io.NDArrayIter(data=val_img)

            predict_results = mod.predict(val)[0][0]
            # print(len(predict_results))
            # background_indices = np.where(predict_results[:, 0].astype(int) > 0)[0]
            # pred = np.delete(predict_results, background_indices, axis=0)

            max_area = 0
            for box in predict_results:
                if box[0] >= 4 and box[1] > 0.4:
                    area = (box[4] - box[2])*(box[5] - box[3])
                    if area > max_area :
                        max_area_box = box
                        max_area = area
                    if box[1] == -1 :
                        break

            np_original_img = np.asarray(img, dtype=np.uint8)
            hei, wid, cha= np_original_img.shape


            xmin = int((max_area_box[2]*wid).asnumpy())
            ymin = int((max_area_box[3]*hei).asnumpy())
            xmax = int((max_area_box[4]*wid).asnumpy())
            ymax = int((max_area_box[5]*hei).asnumpy())

            draw_img = np_original_img.copy()
            cv2.rectangle(draw_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(draw_img, str(max_area_box[1].asnumpy()[0]), (int(xmin),int(ymin)-10), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imwrite(os.path.join(draw_head_path, imgname.split("/")[-1].replace("\n", "")), draw_img)

            # np_head = np_original_img[ymin:ymax, xmin:xmax]
            # img_head = Image.fromarray(np_head.astype('uint8'))
            # img_head.save(os.path.join(head_img_path, imgname.split('/')[-1]))

            # xmin = int(max((max_prob_box[2]*wid).asnumpy()-200, 0))
            # ymin = int(max((max_prob_box[3]*hei).asnumpy()-50, 0))
            # xmax = int(min((max_prob_box[4]*wid).asnumpy()+200, wid))
            # ymax = int(min((max_prob_box[5]*hei).asnumpy()+300, hei))


            head_width = xmax - xmin
            head_height = ymax - ymin
            # assert head_height <= 920, "head height greater than 920"
            # assert head_width <= 920, "head width greater than 920"
            # left_pad = (920 -  head_width)//2
            # right_pad = 920 - head_width - left_pad
            # down_pad = (920 - head_height)//2 
            # up_pad = 920 - head_height - down_pad

            if head_height > head_width:
                left_pad = (head_height -  head_width)//2
                right_pad = head_height - head_width - left_pad
                down_pad = 0
                up_pad = 0  
            elif head_height < head_width:
                down_pad = (head_width - head_height)//2 
                up_pad = head_width - head_height - down_pad
                left_pad = 0
                right_pad = 0
            else:
                left_pad = 0
                right_pad = 0
                down_pad = 0
                up_pad = 0 

            xmin_pad = xmin - left_pad
            xmax_pad = xmax + right_pad
            ymin_pad = ymin - up_pad
            ymax_pad = ymax + down_pad 

            if xmin_pad < 0:
                xmin_pad_zero = 0 - xmin_pad
                left = 0
            else:
                xmin_pad_zero = 0
                left = xmin_pad

            if xmax_pad >= wid:
                xmax_pad_zero = xmax_pad - wid + 1
                right = wid -1
            else:
                xmax_pad_zero = 0
                right = xmax_pad

            if ymin_pad < 0:
                ymin_pad_zero = 0 - ymin_pad
                up = 0
            else:
                ymin_pad_zero = 0
                up = ymin_pad

            if ymax_pad >= hei:
                ymax_pad_zero = ymax_pad - hei + 1
                down = hei - 1
            else:
                ymax_pad_zero = 0
                down = ymax_pad
                                                 
            np_head = np_original_img[int(up):int(down), int(left):int(right)]
            np_head_pad = np.pad(np_head, ((int(ymin_pad_zero), int(ymax_pad_zero)), (int(xmin_pad_zero), int(xmax_pad_zero)), (0, 0)), 'constant', constant_values=(128, 128))

            img_head = Image.fromarray(np_head_pad.astype('uint8'))
            img_head.save(os.path.join(head_img_path, imgname.split('/')[-1]))
            wrtie_str = imgname + "\t" + str(xmin_pad) + "\t" + str(ymin_pad) + "\t" + str(xmax_pad) + "\t" + str(ymax_pad) + "\n"

            fhead.writelines(wrtie_str)
        except Exception as e:
            print(e)
            print(imgname)
            # print(max_prob_box)
            fmiss.writelines(imgname+'\n')
            print(cnt)   
        if cnt%100 ==0:        
            print(cnt)
        cnt += 1
    fmiss.close()
    fhead.close()
   