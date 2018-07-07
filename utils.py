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
from prepro import load_doc_vocab, load_label_vocab, load_user_vocab, load_favs, load_topic_vocab

class Model(nn.Module):
    def __init__(self, topic_vocab_size, doc_vocab_size, nclass, idim=20, hdim = 100, nlayers = 3, ndirections = 1):
        super(Model, self).__init__()
        self.doc_embeds = nn.Embedding(doc_vocab_size, idim, padding_idx=1)
        self.fav_embeds = nn.Embedding(topic_vocab_size, idim, padding_idx=1)

        self.gru_read = nn.GRU(input_size = idim, hidden_size = hdim, num_layers = nlayers, bidirectional = (ndirections == 2), batch_first = True)
        self.gru_unread = nn.GRU(input_size = idim, hidden_size = hdim, num_layers = nlayers, bidirectional = (ndirections == 2), batch_first = True)
        self.gru_fav = nn.GRU(input_size = idim, hidden_size = hdim, num_layers = nlayers, bidirectional = (ndirections == 2), batch_first = True)

        self.fc = nn.Linear(nlayers * ndirections * hdim * 3, nclass)
        #self.fc = nn.Sequential(nn.Linear(nlayers * ndirections * hdim * 3, 1000), nn.Linear(1000, nclass))

        
    def forward(self, fav, read, unread):
        xfav = self.fav_embeds(fav)
        xread = self.doc_embeds(read)
        xunread = self.doc_embeds(unread)
        # out: (seq_len, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        _, hread = self.gru_read(xread)
        _, hunread = self.gru_unread(xunread)
        _, fav = self.gru_fav(xfav)

        #print(xuser.size(), hread.size(), hunread.size())
        h = torch.transpose(torch.cat((hread, hunread, fav), 2), 0, 1).contiguous().view(xfav.size()[0], -1)
        #fc_input = torch.cat((h, xuser), 1)
        #print(fc_input.size())
        #print(fc_input.size())
        #print('done')
        
        return self.fc(h)

def test_forward():
    model = Model(1000,500, 500)
    fav  = torch.LongTensor(np.arange(24).reshape(4,6))
    read = torch.LongTensor(np.arange(20).reshape(4,5))
    unread = torch.LongTensor(np.arange(20).reshape(4,5))
    out = model(fav, read, unread).squeeze()
    print(out.size())


PAD_INDEX = 1
def make_tensor(data, user_vocab, fav_dict, train=True):
    #print(data)
    #users = []
    favs = []
    reads = []
    unreads = []
    labels = []
    maxlen = 0
    maxfavlen = 0
    for batch in data:
        if batch is None:
            if train:
                continue
            else:
                print('ERROR! BATCH is NONE')
                exit(0)
        user, read, unread, label = batch

        maxlen = max(maxlen, len(read))
        maxlen = max(maxlen, len(unread))
        #users.append(user)
        reads.append(sorted(read))
        unreads.append(sorted(unread))
        if train:
            labels.append(label)

        fav = fav_dict[user_vocab.itos[user]]
        if fav is None:
            fav = []
        maxfavlen = max(maxfavlen, len(fav))
        favs.append(fav)

    for i in range(len(reads)):
        reads[i].extend([PAD_INDEX]*(maxlen - len(reads[i])))
        unreads[i].extend([PAD_INDEX]*(maxlen - len(unreads[i])))
    
    for i in range(len(favs)):
        favs[i].extend([PAD_INDEX]*(maxfavlen - len(favs[i])))

    if train:
        return torch.LongTensor(favs).cuda(), torch.LongTensor(reads).cuda(), torch.LongTensor(unreads).cuda(), torch.LongTensor(labels).cuda()
    else:
        return torch.LongTensor(favs).cuda(), torch.LongTensor(reads).cuda(), torch.LongTensor(unreads).cuda()

def save_model(args, model):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    filename = 'best{}.pth'.format(random.randint(1,3))
    torch.save(model.state_dict(), os.path.join(args.model_dir, filename))


def accuracy(output, label, topk=(1,100)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res
