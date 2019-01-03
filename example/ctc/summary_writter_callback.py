# coding: utf-8

class summary_writter_eval_metric(object):
    """Logs cross-entropy and evaluation metrics periodically with SummaryWriter.

    Parameters
    ----------
    sw: SummaryWriter object
    auto_reset: bool
    Reset the evaluation metrics after each log.

    """
    def __init__(self, sw, auto_reset = True):
        self.auto_reset = auto_reset
        self.sw = sw
        self.epoch = 0

    def __call__(self, param):
        """Callback to log the cross_entropy and the accuracy."""
        self.epoch += 1
        if param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            self.sw.add_scalar(tag = name_value[0][0], value = name_value[0][1], global_step = self.epoch)
            # self.sw.add_scalar(tag = name_value[0][0], value = name_value[0][1], global_step = self.epoch)
            # self.sw.add_scalar(tag = name_value[1][0], value = name_value[1][1], global_step = self.epoch)    

        if self.auto_reset:
            param.eval_metric.reset()  


