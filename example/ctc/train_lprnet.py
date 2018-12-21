import mxnet as mx
import lprnet
from ctc_metrics import CtcMetrics,CtcMetrics2
import lstm
import logging
import numpy as np
import cv2
import math

class LPRIter2(mx.io.DataIter):
    def __init__(self, path_imgrec="", batch_size=128, data_shape=(3, 24, 94), path_imglist="",
                 label_width=8, mean_img="",lstm_init_states=None,
                 **kwargs):
        super(LPRIter2, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.rec = mx.io.ImageRecordIter(
            path_imgrec=path_imgrec,
            path_imglist=path_imglist,
            label_width=label_width,
            #mean_img=mean_img,
            data_name='data',
            label_name='label',
            data_shape=(3, 24, 94),
            batch_size=batch_size,
            preprocess_threads=4,
            shuffle=True,
            num_parts=1,
            part_index=0
        )
        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]

        self.provide_data = [('data', (batch_size,data_shape[0], data_shape[1]))] + lstm_init_states
        self.provide_label = [('label', (self.batch_size, label_width))]

        #self.reset()

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
        init_state_names = [x[0] for x in self.init_states]
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
        data = np.zeros((self.batch_size,self.data_shape[0],self.data_shape[1]))

        for i in range(len(buf)):
            data[i] = ((buf[i][0] + buf[i][1] + buf[i][2])/3.0).transpose((1,0))
            #print(buf[i].shape)
            #im = buf[i].transpose((1,2,0))
            '''
            if i==0:
                print(data[i])
                #cv2.imshow("e",data[i])
                cv2.imwrite("res3.bmp",buf[i].transpose((1,2,0)))
                cv2.waitKey(0)'''
        self._batch.data = [mx.nd.array(data)] + self.init_state_arrays
        self.data_names=['data'] + init_state_names
        self.label_names = ['label']
        return True


class LPRIter(mx.io.DataIter):
    def __init__(self, path_imgrec="", batch_size=128, data_shape=(3, 94, 24), path_imglist="",
                 label_width=8, mean_img="",
                 **kwargs):
        super(LPRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape

        self.rec = mx.io.ImageRecordIter(
            path_imgrec=path_imgrec,
            path_imglist=path_imglist,
            label_width=label_width,
            mean_img=mean_img,
            data_name='data',
            label_name='label',
            data_shape=(3,24,94),
            batch_size=batch_size,
            preprocess_threads=4,
            shuffle=False,
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
            data[i] = buf[i].transpose((0,2,1))
            #print(buf[i].shape)
            #im = buf[i].transpose((1,2,0))
            '''
            if i==0:
                print(data[i].shape)
                #cv2.imshow("e",data[i])
                cv2.imwrite("res3.bmp",data[i].transpose((1,2,0)))
                cv2.waitKey(0)'''
        self._batch.data = [mx.nd.array(data)]

        return True

def main():
    '''train = mx.io.ImageRecordIter(
        path_imgrec='/opt/data/plate/train_own.rec',
        path_imglist='/opt/data/plate/resized_plate_train_own.txt',
        label_width=8,
        mean_img='/opt/data/plate/mean_plate.bin',
        data_name='data',
        label_name='label',
        data_shape=(3,94,24),
        batch_size=128,
        preprocess_threads=4,
        shuffle=True,
        num_parts=1,
        part_index=0
    )'''

    '''init_states = lstm.init_states(64, 2, 100)
    symbol = lstm.lstm_unroll(
        num_lstm_layer=2,
        seq_len=94,
        num_hidden=100,
        num_label=79,
        loss_type='ctc')

    train = LPRIter2(
        path_imgrec         = '/opt/data/plate/train_own.rec',
        path_imglist        = '/opt/data/plate/resized_plate_train_own.txt',
        label_width         = 8,
        mean_img            = '/opt/data/plate/mean_plate.bin',
        data_shape          = (94,24),
        batch_size          = 64,
        lstm_init_states=init_states
    )'''

    symbol = lprnet.get_symbol()
    batch_size = 64

    train = LPRIter(
        path_imgrec         = '/opt/data/plate/rec/train.rec',
        path_imglist        = '/opt/data/plate/rec/resized_plate_train.txt',
        label_width         = 8,
        mean_img            = '/opt/data/plate/rec/mean_plate.bin',
        data_shape          = (3,94,24),
        batch_size          = batch_size,
    )

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("train1.log")
    logger.addHandler(fh)

    head = '%(asctime)-15s %(message)s'
    #logging.basicConfig(level=logging.DEBUG, format=head)
    model_prefix = '/opt/models/mxnet/plate/plate_lprnet/lprnet_sgd_wd0.00001'
    begin_epoch = -1

    if begin_epoch >=0:
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, begin_epoch)
    else:
        arg_params, aux_params=None, None


    module = mx.mod.Module(
        symbol,
        data_names=['data'],# 'l0_init_c', 'l0_init_h', 'l1_init_c', 'l1_init_h'],
        label_names=['label'],
        logger=logger,
        context=mx.gpu())

    ''' module = mx.mod.Module(
        symbol,
        data_names=['data'],
        label_names=['label'],
        context=mx.gpu())'''

    metrics = CtcMetrics(24)

    num_epoch = 500
    lr = 0.001

    #step_epochs = [5,10,15,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115]
    step_epochs = range(0,num_epoch,50)
    #print(step_epochs)
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= 0.8
    
    logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)           

    epoch_size = math.ceil(61229 / batch_size)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.8, base_lr=lr)

    module.fit(train_data=train,
               #begin_epoch=begin_epoch,
               #eval_data=data_val,
               # use metrics.accuracy or metrics.accuracy_lcs
               #eval_metric=mx.metric.np(
               #    metrics.accuracy, allow_extra_outputs=True),
               eval_metric=CtcMetrics2(24),
               optimizer='sgd',
               optimizer_params={'learning_rate': lr,
                                 'momentum': 0.9,
                                 'lr_scheduler': lr_scheduler,
                                 'wd': 0.00001,
                                 },
               initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
               arg_params = arg_params,
               aux_params=aux_params,
               num_epoch=num_epoch,
               batch_end_callback=mx.callback.Speedometer(128, 50),
               epoch_end_callback=mx.callback.do_checkpoint(model_prefix),
               )


if __name__ == '__main__':
    main()

