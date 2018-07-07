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
from utils import Model, make_tensor, accuracy, save_model


def get_data_loader(args):
    #train_list = glob.glob(settings.DATA_BIN_DIR + r'\train*.pk')
    #print(train_list)
    #filename = train_list[epoch % len(train_list)]
    filename = args.dataset
    print('train file: ', filename)
    return CCIRDataLoader(filename, batch_size=args.batch_size, shuffle=True)


def test_loader(args):
    loader = get_data_loader(args)
    for data in loader:
        user, read, unread, label = make_tensor(data)
        print(user.size(), read.size(), unread.size(), label.size())
        print(user, read, unread, label)
        break

def save_model(args, model):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    filename = 'best{}.pth'.format(random.randint(1,3))
    torch.save(model.state_dict(), os.path.join(args.model_dir, filename))

def validate(model, val_iter, criterion):
    model.eval()
    val_loss = 0
    corrects = 0
    num = 0
    with torch.no_grad():
        for batch in iter(val_iter):
            #inputs, label = inputs.to(device), label.to(device)
            output = model(batch.text).squeeze()
            val_loss += criterion(output, batch.label.squeeze().float()).item()
            preds = torch.ge(output, 0.5).long()
            corrects += preds.eq(batch.label.squeeze()).sum().item()

            num += len(batch)
            print(num, end='\r')
    return val_loss, corrects, num


def train(args):
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

    print(len(topic_vocab.stoi), len(doc_vocab.stoi), len(label_vocab.stoi))
    model = Model(len(topic_vocab.stoi), len(doc_vocab.stoi), len(label_vocab.stoi)).cuda()
    if args.train_from:
        print('Loading checkpoint: ', args.train_from)
        model.load_state_dict(torch.load(args.train_from))

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_corrects = 0
    nIteration = 0
    numTrain = 0

    #for epoch in range(args.epochs):
    print('dataset: {}'.format(args.dataset))
    loader = get_data_loader(args)
        
    for data in loader:
        nIteration += 1
        model.train()

        #while True:
        favs, read, unread, label = make_tensor(data, user_vocab, fav_dict)
        #print(favs.size(), read.size(), unread.size(), label.size())
        optimizer.zero_grad()
        output = model(favs, read, unread)
        #print(output.size())
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        top1, top100 = accuracy(output, label)
        train_corrects += top100

        numTrain += label.size()[0]
        print('{} {} / {} \tLoss: {:.4f} \tTop100: {:.2f}% ({} / {})  \tTop1: {:.2f}% ({} / {}) \tRunning Top100: {:.2f}%'
            .format(os.path.basename(args.dataset), numTrain, loader.len, loss.item(), 
            100. * top100 / len(data), top100, len(data),
            100. * top1 / len(data), top1, len(data),
            100. * train_corrects / numTrain), end='\r')

        if nIteration % 1000 == 0:
            save_model(args, model)
    
        #if nIteration % 200 == 0:
        #    print('\n')
        #    val_loss, corrects, num = validate(model, val_iter, criterion)
        #    val_loss /= (num / args.batch_size)
        #    print('\nVal Loss: {:.4f}, Val Acc: {}/{} ({:.02f}%)\n'.format(
        #        val_loss, corrects, num, 100. * corrects / num))
    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to put training data', required=False)
    parser.add_argument('--model_dir', type=str, default=r'D:\CCIR2018\data\models', help='Directory to save models', required=False)
    parser.add_argument('--train_from', type=str, default=r'D:\CCIR2018\data\models\best1.pth', help='Directory to save models', required=False)
    parser.add_argument('--dataset', type=str, default=r'G:\ccir2018\train_bin\train2.pk', help='dataset filename', required=False)

    args, unknown = parser.parse_known_args()

    #test_loader(args)
    try:
        train(args)
    except:
        print('exception')
    print('done...')
    exit(0)
    #test_forward()
