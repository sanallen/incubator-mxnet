# -*- coding: utf-8 -*-

import os
import mxnet as mx
import numpy as np
import cv2
import shutil

def iou(x, ys):
    """
    Calculate intersection-over-union overlap
    Params:
    ----------
    x : numpy.array
        single box [xmin, ymin ,xmax, ymax]
    ys : numpy.array
        multiple box [[xmin, ymin, xmax, ymax], [...], ]
    Returns:
    -----------
    numpy.array
        [iou1, iou2, ...], size == ys.shape[0]
    """
    ixmin = np.maximum(ys[:, 0], x[0])
    iymin = np.maximum(ys[:, 1], x[1])
    ixmax = np.minimum(ys[:, 2], x[2])
    iymax = np.minimum(ys[:, 3], x[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih
    uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
        (ys[:, 3] - ys[:, 1]) - inters
    ious = inters / uni
    ious[uni < 1e-12] = 0  # in case bad boxes
    return ious

def plot_rectangle(predict_box, label_box, img_name, img_path, error_img_path, error_img_head_path = None):

    if predict_box.ndim == 1:
        predict_box = predict_box.reshape(1, predict_box.size)

    filename = os.path.join(img_path, img_name)
    img = cv2.imread(filename)
    class_list = ['person', 'bicycle', 'tricycle', 'motobike', 'car', 'bus', 'minibus', 'truck']
    font = cv2.FONT_HERSHEY_COMPLEX
    Thickness_box = 4
    Thickness_text = 1
    height = img.shape[0]
    width = img.shape[1]

    img_head = img.copy() # 用于后面保存预测框内的图片
    img_name = img_name.split('/')[-1]
   
    # 红色画出预测框, 标注出置信度，iou，类别
    ious = iou(label_box[1:5], predict_box[:,2:])
    for j in range(predict_box.shape[0]):
        xmin = int(predict_box[j][2]*width)
        ymin = int(predict_box[j][3]*height)
        xmax = int(predict_box[j][4]*width)
        ymax = int(predict_box[j][5]*height)            
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), Thickness_box)
        text = str((class_list[int(predict_box[j][0])], round(predict_box[j][1],4), round(ious[j],4)))
        cv2.putText(img, text, (xmin,ymax+50), font, 1.5, (0,0,255), Thickness_text)
    
    # 绿色画出真实框, 标注出类别 
    cv2.rectangle(img, (int(label_box[1]*width),int(label_box[2]*height)), 
        (int(label_box[3]*width),int(label_box[4]*height)), (0,255,0), Thickness_box)
    cv2.putText(img, class_list[int(label_box[0])], (int(label_box[1]*width),int(label_box[2]*height)-10), 
        font, 1.5, (0,255,0), Thickness_text) 

    # 裁剪出预测框并保存到指定目录下
    if predict_box.shape[0]==1 and error_img_head_path != None:
        img_head = img_head[ymin:ymax, xmin:xmax]
        cv2.imwrite(error_img_head_path+img_name, img_head)
    
    cv2.imwrite(error_img_path+img_name, img)


def find_wrong_detection(labels, preds, list_path, img_path, ovp_thresh = 0.5):
    """
    compare the labels and preds to find false negative and false positive.

    Params:
    ----------
    labels: mx.nd.array (n * 6) or (n * 5), difficult column is optional
        2-d array of ground-truths, n objects(id-xmin-ymin-xmax-ymax-[difficult])
        labels.shape : test sample number * 1 * 6
        labels.type : <class 'mxnet.ndarray.ndarray.NDArray'>
    preds: mx.nd.array (m * 6)
        2-d array of detections, m objects(id-score-xmin-ymin-xmax-ymax)\
        preds.shape : test sample number * anchor number * 6
        preds.type : <class 'mxnet.ndarray.ndarray.NDArray'>
    该函数只考虑了每张图片中有且只有一个真实框的情景
    """

    flags = [-1]*labels.shape[0]
    # -1： 未设置，背景，真实和预测都为背景
    # 0 ： 正确
    # 1 ： iou<ovpt_hresh
    # 2 ： 预测框的类别数少于真实框个数(漏检)或个数一致但类别不一致

    # 存放类别不一致的错误图片
    wrong_class_img_path  = os.path.join(img_path, 'worng_class/')
    # 存放iou
    low_iou_img_path  = os.path.join(img_path, 'low_iou/')
    low_iou_img_head_path  = os.path.join(img_path, 'low_iou_head/')
    if os.path.exists(wrong_class_img_path):
        shutil.rmtree(wrong_class_img_path)
    os.mkdir(wrong_class_img_path)
    if os.path.exists(low_iou_img_path):
        shutil.rmtree(low_iou_img_path)
    os.mkdir(low_iou_img_path)
    if os.path.exists(low_iou_img_head_path):
        shutil.rmtree(low_iou_img_head_path)
    os.mkdir(low_iou_img_head_path)

    fp = open(list_path)
    listlines = fp.readlines()
    img_name_list = []
    for lines in listlines:
        imgname = lines.split('\t')[-1]
        # 去除换行符
        imgname = imgname.replace('\r','').replace('\n','').replace('\t','')
        img_name_list.append(imgname)
    
    # 存放每张图片预测框的iou最大值，
    iou_list = []
        
    for i in range(labels.shape[0]):
        # get as numpy arrays
        label = labels[i].asnumpy()
        pred = preds[i].asnumpy()
        # assert label.shape[0] == 1  # 确认有且只有一个真实框
        # print (label.shape[0])
        img_name = img_name_list[i]
        # 删除预测为背景的预测框
        background_indices = np.where(pred[:, 0].astype(int) < 0)[0]
        pred = np.delete(pred, background_indices, axis=0)
        if pred.shape[0] == 0:    # 预测框全为背景,即预测框个数少于真实框个数
            flags[i] = 2
            plot_rectangle(pred, label[0], img_name, img_path, wrong_class_img_path)
            iou_list = iou_list + [0]
            continue

        cid = int(pred[0, 0])
        indices = np.where(pred[:, 0].astype(int) == cid)[0]
        pred = pred[indices]
        if label[0,0].astype(int) == cid:
            # 此处iou函数要求第一个参数是单个框，因此必须将label放前面
            ious = iou(label[0][1:5], pred[:, 2:])
            # 选出所有锚点框中iou最大的一个
            ovargmax = np.argmax(ious)
            ovmax = ious[ovargmax]
            iou_list = iou_list + [ovmax]
            if ovmax > ovp_thresh:
                flags[i] = 0  # 位置检测正确  类别一致，iou>threshold
            else:
                flags[i] = 1   # 位置检测错误  类别一致，iou<threshold
                plot_rectangle(pred[ovargmax], label[0], img_name, img_path, low_iou_img_path, low_iou_img_head_path)
        else:
            flags[i] = 2 # 真实框与预测框类别不一致
            plot_rectangle(pred[ovargmax], label[0], img_name, img_path, wrong_class_img_path)
        
    return (flags, iou_list)

