# -*- coding: utf-8 -*-
'''
単純な文章学習用のニューラルモデル
initとcallを変更して好きなネットワークに変更可能
param
n_vocab:入力層及び出力層の数。語彙数。
n_units:隠れ層のノード数。
x_data:単語id
y_data:単語id
train:trueで誤差を出力、falseでスコアを出力
dropout_ratio:ドロップアウト率
'''
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

class normalTweetChainer(chainer.Chain):
    # initialize
    def __init__(self, n_vocab, n_units):
        super(normalTweetChainer, self).__init__(
            embed = L.EmbedID(n_vocab, n_units),
            H1 = L.LSTM(n_units, n_units),
            H2 = L.LSTM(n_units, n_units),
            H3 = L.LSTM(n_units, n_units),
            H4 = L.LSTM(n_units, n_units),
            H5 = L.LSTM(n_units, n_units),
            W = L.Linear(n_units, n_vocab),
            )

    # 順伝搬関数。volatileは学習時はFalseにする必要があります。
    def __call__(self, x_data, y_data, train=True, dropout_ratio=0.5):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)
        h0      = self.embed(x)
        h1      = self.H1(F.dropout(h0, ratio=dropout_ratio, train=train))
        h2      = self.H2(F.dropout(h1, ratio=dropout_ratio, train=train))
        h3      = self.H3(F.dropout(h2, ratio=dropout_ratio, train=train))
        h4      = self.H4(F.dropout(h3, ratio=dropout_ratio, train=train))
        h5      = self.H5(F.dropout(h4, ratio=dropout_ratio, train=train))
        y       = self.W(F.dropout(h5, ratio=dropout_ratio, train=train))
        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.softmax(y)
    def resetState(self):
        self.H1.reset_state()
        self.H2.reset_state()
        self.H3.reset_state()
        self.H4.reset_state()
        self.H5.reset_state()