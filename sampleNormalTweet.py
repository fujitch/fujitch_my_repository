# -*- coding: utf-8 -*-
"""
trainNormalTweetで学習させたモデルからツイートをさせます。
"""

from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import pickle
import numpy as np
from random import randint
import normalTweetChainer

# 初期値設定
vocab_path = './vocab.bin'                # 語彙の辞書のパス
n_units = 512                             # 隠れ層の数
model_path = './seq2seq-80.model'         # ロードするモデルのパス

vocab = pickle.load(open(vocab_path, 'rb'))
n_vocab = len(vocab)
id2wd = {v:k for k, v in vocab.items()}

# モデルの読みこみ
model = normalTweetChainer(n_vocab, n_units)

serializers.load_npz(model_path, model)

# 最初に渡す言葉
key = randint(0, len(vocab)-1)
# key = len(vocab)-1
while True:
    dummy = np.ones((1,1), dtype=np.int32)
    x_data = np.ones((1,1), dtype=np.int32)
    x_data = x_data*key
    key = np.argmax(model.forward_one_step(model, x_data=x_data, y_data=dummy, train=False).data[0])
    word = id2wd[key]
    if word == '<eos>':
        break
    print word
    