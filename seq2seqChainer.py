# -*- coding: utf-8 -*-
"""
対話学習用モデルクラス
param
jv:インプットの語彙数
ev:アウトプットの語彙数
k:隠れ層のノード数
jvocab:入力の語彙の辞書
evocab:出力の語彙の辞書
xp:cpuを使用ならnumpyクラス、gpuを使用ならcuda.cupyクラス
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import chainer.functions as F
import chainer.links as L


class seq2seqChainer(chainer.Chain):
    # initialize
    def __init__(self, jv, ev, k, jvocab, evocab, xp):
        super(seq2seqChainer, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),
            H1 = L.LSTM(k, k),
            H2 = L.LSTM(k, k),
            H3 = L.LSTM(k, k),
            H4 = L.LSTM(k, k),
            H5 = L.LSTM(k, k),
            W = L.Linear(k, ev),
            )

    # 順伝搬関数
    def __call__(self, jline, eline, jvocab, evocab, xp, dropout_ratio=0.5):
        # uttr(発話)を順伝搬させ、LSTMに記憶させる
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(xp.array([wid], dtype=xp.int32)))
            h1 = self.H1(F.dropout(x_k, ratio=dropout_ratio, train=True))
            h2 = self.H2(F.dropout(h1, ratio=dropout_ratio, train=True))
            h3 = self.H3(F.dropout(h2, ratio=dropout_ratio, train=True))
            h4 = self.H4(F.dropout(h3, ratio=dropout_ratio, train=True))
            h5 = self.H5(F.dropout(h4, ratio=dropout_ratio, train=True))
        # 予測スコアを出し、答えとの誤差を足していく
        x_k = self.embedx(Variable(xp.array([jvocab['<eos>']], dtype=xp.int32)))
        tx = Variable(xp.array([evocab[eline[0]]], dtype=xp.int32))
        h1 = self.H1(F.dropout(x_k, ratio=dropout_ratio, train=True))
        h2 = self.H2(F.dropout(h1, ratio=dropout_ratio, train=True))
        h3 = self.H3(F.dropout(h2, ratio=dropout_ratio, train=True))
        h4 = self.H3(F.dropout(h3, ratio=dropout_ratio, train=True))
        h5 = self.H3(F.dropout(h4, ratio=dropout_ratio, train=True))
        accum_loss = F.softmax_cross_entropy(self.W(h5), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(xp.array([wid], dtype=xp.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(xp.array([next_wid], dtype=xp.int32))
            h1 = self.H1(F.dropout(x_k, ratio=dropout_ratio, train=True))
            h2 = self.H2(F.dropout(h1, ratio=dropout_ratio, train=True))
            h3 = self.H3(F.dropout(h2, ratio=dropout_ratio, train=True))
            h4 = self.H3(F.dropout(h3, ratio=dropout_ratio, train=True))
            h5 = self.H3(F.dropout(h4, ratio=dropout_ratio, train=True))
            loss = F.softmax_cross_entropy(self.W(h5), tx)
            accum_loss += loss
        return accum_loss
    # LSTM記憶領域削除
    def resetState(self):
        self.H1.reset_state()
        self.H2.reset_state()
        self.H3.reset_state()
        self.H4.reset_state()
        self.H5.reset_state()
        
    def mt(self, jline, id2wd, jvocab, evocab):
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32), volatile='on'))
            h1 = self.H1(x_k)
            h2 = self.H2(h1)
            h3 = self.H3(h2)
            h4 = self.H4(h3)
            h5 = self.H5(h4)
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32), volatile='on'))
        h1 = self.H1(x_k)
        h2 = self.H2(h1)
        h3 = self.H3(h2)
        h4 = self.H4(h3)
        h5 = self.H5(h4)
        wid = np.argmax(F.softmax(self.W(h5)).data[0])
        if wid in id2wd:
            print id2wd[wid],
        else:
            print wid,
        loop = 0
        while (wid != evocab['<eos>']) and (loop <= 30):
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32), volatile='on'))
            h1 = self.H1(x_k)
            h2 = self.H2(h1)
            h3 = self.H3(h2)
            h4 = self.H4(h3)
            h5 = self.H5(h4)
            wid = np.argmax(F.softmax(self.W(h5)).data[0])
            if wid in id2wd:
                print id2wd[wid],
            else:
                print wid,
            loop += 1
        print