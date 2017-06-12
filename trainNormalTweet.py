# -*- coding: utf-8 -*-

"""
文章作成モデル学習用スクリプト
ツイート文の学習用
１文ずつ改行で区切られたtxtを学習データとして使う。
"""

import codecs
import numpy as np
import pickle
from chainer import cuda, Function, gradient_check, optimizers, serializers, utils
from random import randint
import normalTweetChainer

# データセット作成
datasetText = []
# テキスト読み込み
f = codecs.open('normal.txt', 'r', 'utf-8')
normalText = f.read()
f.close()

# 改行ごとに分ける
normalText = normalText.split('\n')



# 語彙の辞書作成
vocab = {}
for words in datasetText:
    words = words.split('\t')
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
vocab['<eos>'] = len(vocab)
n_vocab = len(vocab)

pickle.dump(vocab, open('./vocab.bin', 'wb'))
        


# 初期値設定
n_units = 512                 # 隠れ層のノード数
batch_size = 10               # バッチサイズ
epochs = 10000                # 学習繰り返し回数
save_roop = 100               # modelを保存する頻度
out_path = 'C:/Users/fujita.FILESERVER2/workspacePy/twitter-master/0605_model'  # 学習モデル出力ディレクトリ

# モデルインスタンス作成
model = normalTweetChainer(len(vocab), n_units)

# gpu使用
'''
gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)
xp = cuda.cupy
''' 
xp = np

# 重みを初期化
for param in model.parameters:
    param[:] = xp.random.uniform(-0.1, 0.1, param.shape)
# 最適化手法を選択
optimizer = optimizers.Adam()
optimizer.setup(model)

# training
for epoch in range(epochs):
    loss = 0
    model.zerograds()
    model.resetState()
    maxLength = 0
    wordsBatchList = []
    # 学習用バッチ作成
    for i in range(batch_size):
        randk = randint(0,len(datasetText)-1)
        words = datasetText[randk]
        words = words.split('\t')
        if len(words) > maxLength:
            maxLength = len(words)
        wordsBatchList.append(words)
    # バッチ内の単語をidに変換し行列に。行：バッチ数、列：文章の長さ数
    keysBatchList = xp.ones((batch_size, maxLength+1), dtype=xp.int32)
    keysBatchList = keysBatchList*(len(vocab)-1)                # 末尾が余るので<eos>で埋める
    for (row, words) in enumerate(wordsBatchList):
        for (col, word) in enumerate(words):
            keysBatchList[row, col] = vocab[word]
    # 順伝搬
    loss = 0
    for i in range(maxLength):
        loss += model(model, x_data=keysBatchList[:,i], y_data=keysBatchList[:,i])
    # 誤差逆伝搬
    loss.backward()
    loss.unchain_backward()
    optimizer.update()
    # 学習save_roop周ごとにモデルを保存
    print str(epoch), " finished", "loss:", str(loss)
    if epoch%10 == 0:
        outfile = "./seq2seq-" + str(epoch) + ".model"
        serializers.save_npz(outfile, model)
