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
"""Contains a class for calculating CTC eval metrics"""

from __future__ import print_function

import numpy as np
import mxnet as mx

class CtcMetrics(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    @staticmethod
    def ctc_label(p):
        """
        Iterates through p, identifying non-zero and non-repeating values, and returns them in a list
        Parameters
        ----------
        p: list of int

        Returns
        -------
        list of int
        """
        ret = []
        p1 = [0] + p
        for i, _ in enumerate(p):
            c1 = p1[i]
            c2 = p1[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        return ret

    @staticmethod
    def _remove_blank(l):
        """ Removes trailing zeros in the list of integers and returns a new list of integers"""
        ret = []
        for i, _ in enumerate(l):
            if l[i] == 0:
                break
            ret.append(l[i])
        return ret

    @staticmethod
    def _lcs(p, l):
        """ Calculates the Longest Common Subsequence between p and l (both list of int) and returns its length"""
        # Dynamic Programming Finding LCS
        if len(p) == 0:
            return 0
        P = np.array(list(p)).reshape((1, len(p)))
        L = np.array(list(l)).reshape((len(l), 1))
        M = np.int32(P == L)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                up = 0 if i == 0 else M[i-1, j]
                left = 0 if j == 0 else M[i, j-1]
                M[i, j] = max(up, left, M[i, j] if (i == 0 or j == 0) else M[i, j] + M[i-1, j-1])
        return M.max()

    def accuracy(self, label, pred):
        """ Simple accuracy measure: number of 100% accurate predictions divided by total number """
        hit = 0.
        total = 0.
        batch_size = label.shape[0]
        for i in range(batch_size):
            l = self._remove_blank(label[i])
            p = []
            #print(pred.shape)
            for k in range(self.seq_len):
                p.append(np.argmax(pred[k * batch_size + i]))
                #p.append(np.argmax(pred[i * self.seq_len + k]))
            p = self.ctc_label(p)
            #if len(p) > 1:
               # print(l,p)
            if len(p) == len(l):
                match = True
                for k, _ in enumerate(p):
                    if p[k] != int(l[k]):
                        match = False
                        break
                if match:
                    hit += 1.0
            total += 1.0
        assert total == batch_size
        return hit / total

    def accuracy_lcs(self, label, pred):
        """ Longest Common Subsequence accuracy measure: calculate accuracy of each prediction as LCS/length"""
        hit = 0.
        total = 0.
        batch_size = label.shape[0]
        for i in range(batch_size):
            l = self._remove_blank(label[i])
            p = []
            for k in range(self.seq_len):
                p.append(np.argmax(pred[k * batch_size + i]))
            p = self.ctc_label(p)
            hit += self._lcs(p, l) * 1.0 / len(l)
            total += 1.0
        assert total == batch_size
        return hit / total

class CtcMetrics2(mx.metric.EvalMetric):
    def __init__(self, seq_len ,eps=1e-8):
        super(CtcMetrics2, self).__init__('CTC')
        self.seq_len = seq_len
        self.eps = eps
        self.num = 2
        self.name = ['acc', 'loss']
        self.sum_metric = [0.0]*self.num
        self.reset()

    def reset(self):
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def reset_local(self):
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

        
    def update(self, labels, preds):
        #acc
        #print(labels[0].asnumpy().shape, preds[0].asnumpy().shape,preds[1].asnumpy().shape)
        #print(preds[1].asnumpy())
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        minibatch = label.shape[0]
        loss = preds[1].asnumpy()
       # print(label.shape)
        #print(self.sum_metric[0])
        for i in range(minibatch):
            l = CtcMetrics._remove_blank(label[i])
            p = []
            #print(pred.shape)
            for k in range(self.seq_len):
                p.append(np.argmax(pred[k * minibatch + i]))
                #p.append(np.argmax(pred[i * self.seq_len + k]))
            p = CtcMetrics.ctc_label(p)
            #if len(p) > 1:
               # print(l,p)
            if len(p) == len(l):
                match = True
                for k, _ in enumerate(p):
                    if p[k] != int(l[k]):
                        match = False
                        break
                if match:
                    self.sum_metric[0] += 1.0
            self.sum_metric[1] += loss[i]
            self.num_inst[0] += 1.0
            self.num_inst[1] += 1.0
        #print(self.sum_metric[0], minibatch)

    def get(self):
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
