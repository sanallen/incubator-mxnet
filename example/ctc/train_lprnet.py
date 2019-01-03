import os
import mxnet as mx
# import lprnet
# import lprnet_tvm
# import lprnet_paper
import lprnet_concat
# import lprnet_fc
import lprnet_res
from ctc_metrics import CtcMetrics,CtcMetrics2
import lstm
import logging
import numpy as np
import cv2
import math
from mxboard import SummaryWriter
import summary_writter_callback



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
                # min_aspect_ratio    = None,
                brightness          = 0.1,
                contrast            = 0.1,
                saturation          = 0.1,
                random_h            = 9,
                # random_s            = 10,
                # random_l            = 10,
                max_rotate_angle    = 5,
                max_shear_ratio     = 0.1,
                pca_noise           = 0.05,
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
            data[i] = buf[i].transpose((0,2,1))
        self._batch.data = [mx.nd.array(data)]

        return True

def main():

    # symbol = lprnet.get_symbol()
    # symbol = lprnet_tvm.get_symbol()
    # symbol = lprnet_paper.get_symbol()
    symbol = lprnet_concat.get_symbol()
    # symbol = lprnet_fc.get_symbol()
    # symbol = lprnet_res.get_symbol()
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

    val = LPRIter(
        path_imgrec         = '/opt/data/plate/rec/test.rec',
        path_imglist        = '/opt/data/plate/rec/resized_plate_test.txt',
        label_width         = 8,
        mean_img            = '/opt/data/plate/rec/mean_plate.bin',
        data_shape          = (3,94,24),
        batch_size          = batch_size,
        augment             = False
    )

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("lprnet_sgd_concat4")
    # logging.info('use augment %d' % augment)
    logger.addHandler(fh)
    log_dir = '/opt/incubator-mxnet/example/ctc/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # sw = SummaryWriter(logdir=log_dir, flush_secs = 180) # 180s
    # sw.add_graph(symbol)

    head = '%(asctime)-15s %(message)s'
    #logging.basicConfig(level=logging.DEBUG, format=head)
    model_prefix = '/opt/models/mxnet/plate/plate_lprnet/lprnet_sgd_concat4'
   
    begin_epoch = 0 # 填写预训练模型的epoch

    if begin_epoch >0:
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, begin_epoch)
    else:
        arg_params, aux_params=None, None


    module = mx.mod.Module(
        symbol,
        data_names=['data'],# 'l0_init_c', 'l0_init_h', 'l1_init_c', 'l1_init_h'],
        label_names=['label'],
        logger=logger,
        context=mx.gpu())


    metrics = CtcMetrics(24)

    num_epoch = 500
    lr = 0.001
    logging.info('Initinal learning rate \"%f\"', lr)
    #step_epochs = [5,10,15,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115]
    # step_epochs = range(0,num_epoch,50)
    step_epochs = range(0,num_epoch,50)
    # step_epochs = [80,120,160,65,70,75,80,85,90,95,100,105,110,115]
    #print(step_epochs)
    for s in step_epochs[1:]:
        if begin_epoch >= s:
            lr *= 0.5
    
    logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)           

    epoch_size = math.ceil(61229 / batch_size)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs[1:] if x - begin_epoch > 0]

    logging.info('learning rate before training\"%f\"', lr)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=0.5, base_lr=lr)

    wd = 0.0005
    logging.info('weight decay \"%f\"', wd)
    # batch_end_callbacks = [mx.callback.Speedometer( 
    #     128, 50, False), summary_writter_callback.summary_writter_eval_metric(sw)]
    batch_end_callbacks = [mx.callback.Speedometer( 
        128, 50, True)]
    module.fit(train_data=train,
               eval_data=val,
               eval_metric=CtcMetrics2(24),
            #    eval_metric=CtcMetrics(24),
            #    eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
               optimizer='sgd',
               optimizer_params={'learning_rate': lr,
                                 'momentum': 0.9,
                                 'lr_scheduler': lr_scheduler,
                                 'wd': wd,
                                 },
               initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
               arg_params = arg_params,
               aux_params=aux_params,
               begin_epoch=begin_epoch,
               num_epoch=num_epoch,
               validation_metric=CtcMetrics2(24),
               batch_end_callback=batch_end_callbacks,
               epoch_end_callback=mx.callback.do_checkpoint(model_prefix),
               )


if __name__ == '__main__':
    main()

