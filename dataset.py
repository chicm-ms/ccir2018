import numpy as np
import os
import torch
import torch.utils.data as data
import pickle
import settings

class CCIRDataLoader(object):
    def __init__(self, filename, shuffle=True, batch_size=8):
        self.shuffle = shuffle
        self.batch_size = batch_size

        with open(filename, 'rb') as f:
            self.dataset = pickle.load(f)
        self.len = len(self.dataset)

        if shuffle:
            self.sampler = torch.utils.data.sampler.RandomSampler(self.dataset) 
            drop_last = True
        else:
            self.sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            drop_last = False

        self.batch_sampler = torch.utils.data.sampler.BatchSampler(self.sampler, batch_size, drop_last)

        self.sample_iter = iter(self.batch_sampler)

    def __iter__(self):
        return self
    
    def __next__(self):
        indices = next(self.sample_iter)
        #print(indices)
        return [self.dataset[i] for i in indices]

def test_dataset():
    print('testing')
    val_loader = CCIRDataLoader(os.path.join(settings.DATA_BIN_DIR, 'train1.pk'), batch_size=4, shuffle=True)
    print(val_loader.len)
    for data in val_loader:
        print(data)
        break
    
if __name__ == '__main__':
    test_dataset()