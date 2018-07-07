import os
import glob
import argparse
import re
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import settings
from dataset import CCIRDataLoader
from utils import Model, make_tensor
from prepro import load_doc_vocab, load_label_vocab, load_user_vocab, load_favs, load_topic_vocab

CP = r'D:\ccir2018\data\models\best1.pth'

def predict():
    print('Loading user vocab...')
    user_vocab = load_user_vocab()
    print('Loading doc vocab...')
    doc_vocab = load_doc_vocab()
    print('Loading label vocab...')
    label_vocab = load_label_vocab()
    print('Loading favs...')
    fav_dict = load_favs()
    print('Loading topic vocab...')
    topic_vocab = load_topic_vocab()

    filename = os.path.join(settings.TEST_DATA_DIR, 'test.pk')
    print('test file: ', filename)
    loader = CCIRDataLoader(filename, batch_size=64, shuffle=False)

    print(len(topic_vocab.stoi), len(doc_vocab.stoi), len(label_vocab.stoi))
    model = Model(len(topic_vocab.stoi), len(doc_vocab.stoi), len(label_vocab.stoi)).cuda()
    print('Loading checkpoint: ', CP)
    model.load_state_dict(torch.load(CP))
    model.eval()

    m = nn.Softmax()

    preds = None

    for data in loader:
        favs, read, unread = make_tensor(data, user_vocab, fav_dict, train=False)
        output = m(model(favs, read, unread))
        _, pred = output.topk(100, 1, True, True)
        pred = np.array(pred.cpu().tolist())

        if preds is None:
            preds = pred
        else:
            preds = np.vstack((preds, pred))
    print(preds.shape)
    np.save(os.path.join(settings.DATA_BIN_DIR, 'preds.npy'), preds)

def make_submit():
    label_vocab = load_label_vocab()
    preds = np.load(os.path.join(settings.DATA_BIN_DIR, 'preds.npy'))
    
    print(preds[0])

    submits = []
    for pred in preds:
        longids = [label_vocab.itos[i] for i in pred]
        submit_line = [ x[:4]+x[-4:] for x in longids]
        #print(submit_line)
        assert(len(set(submit_line)) == 100)
        submits.append(submit_line)

    with open(os.path.join(settings.DATA_BIN_DIR, 'sub1.txt'), 'w') as f:
        for line in submits:
            f.write(','.join(line)+'\n')


if __name__ == '__main__':
    make_submit()