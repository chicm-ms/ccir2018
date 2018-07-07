import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets

class Model(nn.Module):
    def __init__(self, vocab_size, dim = 100, nlayers = 15):
        super(Model, self).__init__()
        self.embeds = nn.Embedding(vocab_size, dim, padding_idx=1)
        self.gru = nn.GRU(input_size = dim, hidden_size = dim, num_layers = nlayers, bidirectional=True, dropout = 0.5)
        self.fc = nn.Linear(nlayers * 2 * dim, 5)
    
    def forward(self, inputs):
        x = self.embeds(inputs)
        _, hidden = self.gru(x)
        return self.fc(torch.transpose(hidden, 0, 1).contiguous().view(x.size(1), -1))
        #return self.fc(hidden.view(x.size(1), -1))

def test_forward():
    model = Model(1000).cuda()
    inputs = torch.LongTensor(np.arange(20).reshape(5,4)).cuda()
    out = model(inputs)
    print(out.size())

def get_sst_iter(args):
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)
    train, val, test = datasets.SST.splits(TEXT, LABEL, fine_grained=True, train_subtrees=True,
        filter_pred=lambda ex: ex.label != 'neutral')
    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('len(test)', len(test))
    print('len(val)', len(val))
    print('vars(train[1])', vars(train[1]))
    print('vars(test[1])', vars(test[1]))
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    LABEL.vocab.stoi = { w: i for i, w in enumerate(labels) }
    LABEL.vocab.itos = { i: labels[i] for i in range(len(labels)) }
    print(LABEL.vocab.stoi)
    print(LABEL.vocab.itos)
    print('<pad>:', TEXT.vocab.stoi['<pad>'])
    print('len(TEXT.vocab)', len(TEXT.vocab))
    #print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size)
    return train_iter, val_iter, test_iter, TEXT.vocab

# make splits for data
#train, test = datasets.IMDB.splits(TEXT, LABEL)



def save_model(args, model, filename):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.state_dict(), os.path.join(args.model_dir, filename))

def validate(model, val_iter, criterion):
    model.eval()
    val_loss = 0
    corrects = 0
    num = 0
    with torch.no_grad():
        for batch in iter(val_iter):
            #inputs, label = inputs.to(device), label.to(device)
            output = model(batch.text)
            val_loss += criterion(output, batch.label).item()
            preds = output.max(1, keepdim=True)[1]
            corrects += preds.eq(batch.label.view_as(preds)).sum().item()

            num += len(batch.label)
            print(num, end='\r')
    return val_loss, corrects, num

def show_batch(batch, text_vocab):
    ix = torch.transpose(batch.text, 0, 1)
    for index in ix:
        sentence = [text_vocab.itos[i] for i in index]
        print(sentence)


def train(args):
    train_iter, val_iter, test_iter, vocab = get_sst_iter(args)
    model = Model(len(vocab)).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    best_corrects = 0
    nIteration = 0
    numTrain = 0

    for batch in iter(train_iter):
        nIteration += 1
        model.train()
        #label = torch.index_select(torch.eye(5, dtype=torch.long).cuda(), 0, batch.label)
        optimizer.zero_grad()
        output = model(batch.text)
        #print(output.size(), label.size())
        #print(batch.label)
        
        loss = criterion(output, batch.label)
        loss.backward()
        optimizer.step()
        numTrain += len(batch)

        preds = output.max(1, keepdim=True)[1]
        corrects = preds.eq(batch.label.view_as(preds)).sum().item()

        print('Epoch: {:.2f} \tLoss: {:.4f} \t Acc: {}/{}'.format(numTrain / len(train_iter.dataset), loss.item(), corrects, len(batch)), end='\r')

        if nIteration % 200 == 0:
            val_loss, corrects, num = validate(model, val_iter, criterion)
            val_loss /= (num / args.batch_size)
            print('\nValidation loss: {:.4f}, Validation accuracy: {}/{} ({:.02f}%)\n'.format(
                val_loss, corrects, num, 100. * corrects / num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate", required=False)
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size", required=False)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs', required=False)
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to put training data', required=False)
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save models', required=False)

    args, unknown = parser.parse_known_args()

    train(args)
