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

import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import logging
import os
import cv2

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=self.ctx)
        if not isinstance(data_shape, tuple):
            data_shape = (data_shape, data_shape)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[0], data_shape[1]))])
        self.mod.set_params(args, auxs)
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        print(detections.shape)
        time_elapsed = timer() - start
        if show_timer:
            logging.info("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        result = Detector.filter_positive_detections(detections)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)
    
    def save_detection(self, img, dets, classes=[], thresh=0.6, outpath = ''):
        height = img.shape[0]
        width = img.shape[1]
        list_path = os.path.join(outpath.rsplit("/", 2)[0], "car_head_train.txt")
        img_name = outpath.rsplit("/", 1)[1].replace('txt', 'jpg')

        detfile = open(list_path, 'a+')
        if len(dets[0])==0:
            return
        for det in dets:
            (klass, score, x0, y0, x1, y1) = det

            if score < thresh:
                continue
            cls_id = int(klass)
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)

            head_width = xmax - xmin
            head_height = ymax - ymin

            left_pad = 0
            right_pad = 0
            down_pad = 0
            up_pad = 0 
            if head_height > head_width:
                left_pad = (head_height -  head_width)//2
                right_pad = head_height - head_width - left_pad 

            if head_height < head_width:
                down_pad = (head_width - head_height)//2 
                up_pad = head_width - head_height - down_pad

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

            if xmax_pad >= width:
                xmax_pad_zero = xmax_pad - width + 1
                right = width -1
            else:
                xmax_pad_zero = 0
                right = xmax_pad

            if ymin_pad < 0:
                ymin_pad_zero = 0 - ymin_pad
                up = 0
            else:
                ymin_pad_zero = 0
                up = ymin_pad

            if ymax_pad >= height:
                ymax_pad_zero = ymax_pad - height + 1
                down = height - 1
            else:
                ymax_pad_zero = 0
                down = ymax_pad

            # head_img = img[xmin:xmax, ymin:ymax]
            # head_img = img[ymin:ymax, xmin:xmax]
            np_head = img[int(up):int(down), int(left):int(right)]
            np_head_pad = np.pad(np_head, ((int(ymin_pad_zero), int(ymax_pad_zero)), (int(xmin_pad_zero), 
                int(xmax_pad_zero)), (0, 0)), 'constant', constant_values=(128, 128))

            # head_img = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(os.path.join(outpath.rsplit("/", 1)[0], img_name), np_head_pad)
            # class_name = str(cls_id)
            # if classes and len(classes) > cls_id:
            # 	class_name = classes[cls_id]
            detfile.write('{} {} {} {} {} {}\n'.format(img_name, xmin_pad, ymin_pad, xmax_pad, ymax_pad, score))
        detfile.close()
    
    def visualize_detection(self, img, dets, classes=[], thresh=0.6, outpath = ''):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        # import matplotlib.pyplot as plt
        import random
        # plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        detnum = 0	
        if len(dets[0]) == 0:
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(os.path.dirname(outpath), 'no_det', os.path.basename(outpath)))
            return	
        for det in dets:
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh:
                continue
            detnum += 1
            cls_id = int(klass)
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)

            cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (0,0,255), 2)
            class_name = str(cls_id)
            if classes and len(classes) > cls_id:
                class_name = classes[cls_id]
            cv2.putText(img, '%s %f'%(class_name,round(score,3)), (xmin,ymin+20),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv2.LINE_AA)	

        if detnum >= 1:

            cv2.imencode('.jpg', img)[1].tofile(os.path.join(os.path.dirname(outpath), 'det', os.path.basename(outpath)))
        else:
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(os.path.dirname(outpath), 'no_det', os.path.basename(outpath)))


    @staticmethod
    def filter_positive_detections(detections):
        """
        First column (class id) is -1 for negative detections
        :param detections:
        :return:
        """
        class_idx = 0
        assert(isinstance(detections, mx.nd.NDArray) or isinstance(detections, np.ndarray))
        detections_per_image = []
        # for each image
        for i in range(detections.shape[0]):
            result = []
            det = detections[i, :, :]

            for obj in det:
                # 去除非机动车预测框
                if obj[class_idx] >= 4:
                    # print(obj[class_idx])
                    result.append(obj)
            detections_per_image.append(result)
            #print(len(result))
        logging.info("%d positive detections", len(result))
        
        return detections_per_image

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False, outdir=''):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        
        no_det_dir = outdir + '/' + 'no_det'
        if not os.path.exists(no_det_dir):
            os.makedirs(no_det_dir)
        
        det_dir = outdir + '/' + 'det'
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
            
        res_dir = outdir + '/' + 'head'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        for k, det in enumerate(dets):
            # img = cv2.imread(im_list[k])
            img = cv2.imdecode(np.fromfile(im_list[k],dtype=np.uint8),-1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            max_area_box = []
            max_area = 0 
            for box in det:
                (klass, score, x0, y0, x1, y1) = box
                if score < thresh:
                    continue
                area = (x1 - x0)*(y1 - y0)
                if area > max_area:
                    max_area = area
                    max_area_box = box
                
            max_det = [max_area_box]
            ori_img = img.copy()
            if outdir != '':
                print('save:', k, im_list[k])
                self.visualize_detection(img, max_det, classes, thresh, os.path.join(outdir,os.path.basename(im_list[k])))
                # self.save_detection(img, max_det, classes, thresh, os.path.join(res_dir,os.path.basename(im_list[k])[:-4]+'.txt'))
                self.save_detection(ori_img, max_det, classes, thresh, os.path.join(res_dir, os.path.basename(im_list[k])[:-4]+'.txt'))
            else:
                self.visualize_detection(img, max_det, classes, thresh)			
            
            
            
            
            
            
            
            
            
            
