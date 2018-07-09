import os
import settings
from collections import Counter
import pickle

def scan_seq_lengths(filename):
    lens = set([])
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    for data in dataset:
        if not data is None:
            if not data[1] is None:
                lens.add(len(data[1]))
            if not data[2] is None:
                lens.add(len(data[2]))
    print(sorted([i for i in lens]))


def scan_fav_lengths():
    lens = set([])
    filename = os.path.join(settings.VOCAB_DIR, 'favs.pk')
    with open(filename, 'rb') as f:
        favs = pickle.load(f)
    for k in favs.keys():
        lens.add(len(favs[k]))
    
    print(sorted([i for i in lens]))

def most_common_fav_lengths(limit):
    counter = Counter()
    filename = os.path.join(settings.VOCAB_DIR, 'favs.pk')
    with open(filename, 'rb') as f:
        favs = pickle.load(f)
    print(len(favs))
    for k in favs.keys():
        counter.update([len(favs[k])])
    
    totalNum = 0
    num = 0
    for elem, count in counter.items():
        totalNum += count
        if elem <= limit:
            num += count

    print('{}/{}, {}%'.format(num, totalNum, 100.*num / totalNum))

def scan(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        count = 0
        for line in f:
            if count < 10:
                print(count)
                print(line)
            else:
                break
            count += 1

#scan(r'F:\competition\training_set.txt')
#scan(os.path.join(settings.DATA_DIR, 'user_infos.txt'))
#scan(os.path.join(settings.TEST_DATA_DIR, 'testing_set_135089.txt'))
#scan_seq_lengths(os.path.join(settings.TEST_DATA_DIR, 'test.pk'))
#scan_seq_lengths(os.path.join(settings.TRAIN_DIR, 'train16.pk'))
#scan_fav_lengths()
#most_common_fav_lengths(1000)
#most_common_fav_lengths(500)
most_common_fav_lengths(160)