# -*- coding: utf-8 -*-
# 根据test_error_list获得测试集（test_image_path）中识别错误的车牌图像保存到error_image_path中
import os
import sys
import shutil

# test_image_path = '/opt/data/plate/adjustment_94x24'
test_image_path = '/opt/data/plate/vpr_yellow_voc/adjustment_94x24'
test_list = '/opt/incubator-mxnet/example/ctc/test_error_list.txt'
# test_list = '/opt/PlateRecognition/LPRnet/test_error_list_backup.txt'
error_image_path = '/opt/data/plate/plate_error_yellow/'
flist = open(test_list,'r')
if os.path.exists(error_image_path):
    shutil.rmtree(error_image_path)
os.mkdir(error_image_path)
# labels = []
# preds = []
label_pred_dicts = {}
# cnt = 0
for filename in flist.readlines():
    parts = filename.split('\t')
    label = parts[0].split('(')[0].strip('_')
    pred = parts[1].split('(')[0].strip('_')
    if label in label_pred_dicts.keys():
        print("repeat label %s" % label)
    label_pred_dicts[label] = pred

error_file_list = []    
for rootpath, dirnames, filenames in os.walk(test_image_path):
    for imagename in filenames:
        # label = imagename.split('_')[-1].split('(')[1].split('-')[0] # label_pred
        # label = imagename.split('~')[-1].split('.')[0] # train
        label = imagename.split('(')[1].split('-')[0]
        if  label in label_pred_dicts.keys():
            pred = label_pred_dicts[label]
            error_file_list.append(os.path.join(rootpath, imagename))
            shutil.copy(os.path.join(rootpath, imagename), os.path.join(error_image_path, label+'_'+pred))

# for path in error_file_list:
#     print(path)
