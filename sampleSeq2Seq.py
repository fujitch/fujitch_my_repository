#!/usr/bin/env python                                                                                                                                                    
# -*- coding: utf-8 -*-                                                                                                                                                  
"""
trainSeq2Seq.pyで学習したモデルseq2seqchainerを使って、インプットされた発話に対する反応を出力するスクリプト
"""
import numpy as np
from janome.tokenizer import Tokenizer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from seq2seqChainer import seq2seqChainer
import pickle


# 初期値設定
mpath = "6_12_model/seq2seq-1.model"              # 学習済みモデルディレクトリ
utt_vocab_path = "jvocab.bin"                     # 発話用語彙辞書
res_vocab_path = "evocab.bin"                     # 反応用語彙辞書
demb = 512                                        # model隠れ層のノード数
utterance = u"こんにちは"                             # インプット

# 語彙の辞書読み込み
jvocab = pickle.load(open(utt_vocab_path, 'rb'))
evocab = pickle.load(open(res_vocab_path, 'rb'))
jv = len(jvocab)
ev = len(evocab)
id2wd = {}
for word in evocab:
    id2wd[evocab[word]] = word

# モデル読み込み
model = seq2seqChainer(jv, ev, demb, jvocab, evocab, np)
serializers.load_npz(mpath, model)
# インプットを形態素解析
t = Tokenizer()
tokens = t.tokenize(utterance)
seq = []
for token in tokens:
    seq.append(token.surface)
jln = seq
jlnr = jln[::-1]                         # 逆向きにする
model.mt(jlnr, id2wd, jvocab, evocab)