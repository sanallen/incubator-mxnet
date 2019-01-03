import mxnet as mx
import numpy as np
import random
import cv2
import Augment

class LPRIter(mx.io.DataIter):
    def __init__(self, path_imgrec="", batch_size=128, data_shape=(3, 94, 24), path_imglist="",
                 label_width=8, mean_img="", augment = True,
                 **kwargs):
        super(LPRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape

        if augment:
            self.rec = mx.io.ImageRecordIter(
                path_imgrec=path_imgrec,
                path_imglist=path_imglist,
                label_width=label_width,
                mean_img=mean_img,
                data_name='data',
                label_name='label',
                data_shape=(3,24,94),
                batch_size=batch_size,
                # max_aspect_ratio    = 0.25,
                min_aspect_ratio    = None,
                brightness          = 0.1,
                contrast            = 0.1,
                saturation          = 0.1,
                random_h            = 9,
                # random_s            = 10,
                # random_l            = 10,
                max_rotate_angle    = 5,
                max_shear_ratio     = 0.1,
                preprocess_threads=8,
                shuffle=True,
                num_parts=1,
                part_index=0
            )
        else:
            self.rec = mx.io.ImageRecordIter(
                path_imgrec=path_imgrec,
                path_imglist=path_imglist,
                label_width=label_width,
                mean_img=mean_img,
                data_name='data',
                label_name='label',
                data_shape=(3,24,94),
                batch_size=batch_size,
                preprocess_threads=8,
                shuffle=True,
                num_parts=1,
                part_index=0
            )

        self.provide_data = [('data', (batch_size,data_shape[0], data_shape[1], data_shape[2]))]
        self.provide_label = [('label', (self.batch_size, label_width))]

        self.reset()

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False
        # modify label
        label = self._batch.label[0].asnumpy()
        
        for i in range(len(label)):
            #if i==0:
                #print(label[i])
            for j in range(len(label[i])):
                if label[i][j] == 78:
                    label[i][j] = 0
                else:
                    label[i][j] += 1
        self._batch.label = [mx.nd.array(label)]

        buf = self._batch.data[0].asnumpy()
        data = np.zeros((self.batch_size,self.data_shape[0],self.data_shape[1],self.data_shape[2]))

        for i in range(len(buf)):
            im = buf[i] # 
            augment_type = random.randint(0,2)
            ori_im = im.transpose(1,2,0).astype(np.uint8)
            im = cv2.imread("/opt/data/plate/im.jpg")
            # if augment_type == 0:               
            # im_Distort = Augment.GenerateDistort(im, 12)
            # cv2.imwrite("/opt/data/plate/im_Distort.jpg", im_Distort)
            # if augment_type == 1:
            im_Stretch = Augment.GenerateStretch(im, 12)
            cv2.imwrite("/opt/data/plate/im_Stretch.jpg", im_Stretch)
            # if augment_type == 2:
            im_Perspective = Augment.GeneratePerspective(im)
            cv2.imwrite("/opt/data/plate/im_Perspective.jpg", im_Perspective)
            data[i]= im.transpose((0,2,1))
            # cv2.imwrite("/opt/data/plate/im.jpg", im.transpose((1,2,0)))
            
            '''
            if i==0:
                print(data[i].shape)
                #cv2.imshow("e",data[i])
                cv2.imwrite("res3.bmp",data[i].transpose((1,2,0)))
                cv2.waitKey(0)'''
        self._batch.data = [mx.nd.array(data)]

        return True

if __name__ == "__main__":

    batch_size = 128
    train = LPRIter(
        path_imgrec         = '/opt/data/plate/rec/train.rec',
        path_imglist        = '/opt/data/plate/rec/resized_plate_train.txt',
        label_width         = 8,
        mean_img            = '/opt/data/plate/rec/mean_plate.bin',
        data_shape          = (3,94,24),
        batch_size          = batch_size,
        augment             = True
    )

    for epoch in range(0, 1):
        data_iter = iter(train)
        end_of_batch = False
        next_data_batch = next(data_iter)
        data_batch = next_data_batch