import os
import sys
import random
import pickle
from collections import Counter
from torchtext.vocab import Vocab
import settings

VOCAB_DIR = settings.VOCAB_DIR

def load_dict(dict, id_file, prefix):
    with open(os.path.join(settings.DATA_DIR, id_file), 'r') as f:
        for line in f:
            s, l = line.strip().split('\t')
            dict[l] = prefix + s

def build_longid_dict():
    long2short = {}
    load_dict(long2short, 'answer_id.dict', 'A')
    load_dict(long2short, 'question_id.dict', 'Q')
    return long2short

def check_candidate_longid():
    long2short = build_longid_dict()
    with open(os.path.join(settings.DATA_DIR, 'candidate.txt'), 'r') as f:
        count = 0
        for line in f:
            _, l = line.strip().split('\t')
            if l in long2short:
                count += 1
        print('found:', count)

def build_doc_vocab(output_file):
    prefix_dict = {'answer_id.dict': 'A', 'question_id.dict': 'Q'}
    counts = Counter()
    for filename in prefix_dict.keys():
        with open(os.path.join(settings.DATA_DIR, filename), 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            for line in lines:
                counts.update([ prefix_dict[filename] + line.split('\t')[0] ])
    vocab = Vocab(counts)
    with open(output_file, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

def random_pick_train_lines(nlines):
    result = []
    line_numbers = set()
    for i in range(nlines):
        line_numbers.add(random.randint(0, 10000))
    max_number = max(line_numbers)
    with open(os.path.join(settings.DATA_DIR, 'training_set.txt'), 'r', encoding='UTF-8') as f:
        count = 0
        for line in f:
            if count in line_numbers:
                result.append(line.strip())
            count += 1
            if count > max_number:
                break
    return result

def build_user_vocab(output_file):
    counts = Counter()
    with open(os.path.join(settings.DATA_DIR, 'user_infos.txt'), 'r', encoding='UTF-8') as f:
        for line in f:
            counts.update([ line.strip().split('\t')[0] ])
    vocab = Vocab(counts)
    with open(output_file, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

def build_topic_vocab(output_file):
    counts = Counter()
    with open(os.path.join(settings.DATA_DIR, 'topic_info.txt'), 'r', encoding='UTF-8') as f:
        for line in f:
            counts.update([ line.strip().split('\t')[0].strip() ])
    vocab = Vocab(counts)
    with open(output_file, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

def build_label_vocab(output_file):
    counts = Counter()
    with open(os.path.join(settings.DATA_DIR, 'candidate.txt'), 'r', encoding='UTF-8') as f:
        for line in f:
            counts.update([ line.strip().split('\t')[1] ])
    vocab = Vocab(counts, specials=[])
    with open(output_file, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

def build_favoriates(output_file, topic_vocab):
    fav_dict = {}
    with open(os.path.join(settings.DATA_DIR, 'user_infos.txt'), 'r', encoding='UTF-8') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            user = fields[0].strip()
            favs = fields[-1]
            if len(favs) < 0:
                fav_dict[user] = []
                continue
            favs = favs.split(',')
            favs = sorted([topic_vocab.stoi[k] for k in favs])
            #print(favs)
            fav_dict[user] = favs
    with open(output_file, 'wb') as f:
        pickle.dump(fav_dict, f)
    return fav_dict

def get_train_indices_old(line, stoi, user_vocab, label_vocab):
    line_arr = line.strip().split('\t')
    if line_arr[1] == '0':
        return None
    seq = line_arr[2].split(',')
    read_seq = [item.split('|')[0] for item in seq if item.split('|')[2] != '0']
    unread_seq = [item.split('|')[0] for item in seq if item.split('|')[2] == '0']
    #print(line)
    #print(len(unread_seq), len(read_seq), line.split('\t')[1])

    label = label_vocab.stoi[line_arr[-1]]
    return user_vocab.stoi[line_arr[0]], [stoi[item] for item in read_seq], [stoi[item] for item in unread_seq], label

def get_train_indices(line, stoi, user_vocab, label_vocab):
    fields = line.strip().split('\t')
    if not len(fields) == 8:
        raise ValueError('Num of fields: {}, Error line:{}'.format(len(fields), line))
    if not int(fields[1]) >= 0:
        raise ValueError('field 1: {}'.format(fields[1]))
    
    if fields[1] == '0':
        seq = []
    else:
        seq = fields[2].split(',')
    
    read_seq = [item.split('|')[0] for item in seq if item.split('|')[2] != '0']
    unread_seq = [item.split('|')[0] for item in seq if item.split('|')[2] == '0']

    label = label_vocab.stoi[fields[-1]]
    if label == 0:
        raise ValueError('Lable not found: [{}]'.format(fields[-1]))
    return user_vocab.stoi[fields[0]], [stoi[item] for item in read_seq], [stoi[item] for item in unread_seq], label

def get_test_indices(line, stoi, user_vocab):
    line_arr = line.strip().split('\t')
    if line_arr[1] == '0':
        read_seq = []
        unread_seq = []
    else:
        seq = line_arr[2].split(',')
        read_seq = [item.split('|')[0] for item in seq if item.split('|')[2] != '0']
        unread_seq = [item.split('|')[0] for item in seq if item.split('|')[2] == '0']
    if line_arr[3] == '0':
        search_seq = []
    else:
        search = line_arr[4].split(',')
        search_seq = [item.split('|')[0].strip() for item in search]
    print(search_seq)

    return user_vocab.stoi[line_arr[0]], [stoi[item] for item in read_seq], [stoi[item] for item in unread_seq], search_seq

def build_train_sequence(output_dir, doc_vocab, user_vocab, label_vocab):
    print('building dataset into: ', output_dir)
    train_file_len = 1000000
    result = []
    error_num = 0
    file_no = 1
    with open(os.path.join(settings.DATA_DIR, 'training_set.txt'), 'r', encoding='UTF-8') as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print(i)
            try:
                res = get_train_indices(line, doc_vocab.stoi, user_vocab, label_vocab)
                result.append(res)
            except:
                info = sys.exc_info()  
                print(info[0], ":", info[1])

                error_num += 1
                print('error:', error_num)
                continue

            if i % train_file_len == 0:
                output_file = os.path.join(output_dir, 'train{}.pk'.format(file_no))
                print(i, 'saving {} records to {}'.format(len(result), output_file))
                with open(output_file, 'wb') as fres:
                    pickle.dump(result, fres)
                    fres.flush()
                    result = []
                    file_no += 1
    
    output_file = os.path.join(output_dir, 'train{}.pk'.format(file_no))
    print('saving {} records to {}'.format(len(result), output_file))
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
        f.flush()
        result = []


def build_train_sequence_old(output_file, doc_vocab, user_vocab, label_vocab, min_index = 0, max_index = 1000000):
    result = []
    error_num = 0
    with open(os.path.join(settings.DATA_DIR, 'training_set.txt'), 'r', encoding='UTF-8') as f:
        i = 0
        for line in f:
            if i < min_index:
                i += 1
                continue
            if i > max_index:
                break
            try:
                res = get_train_indices_old(line, doc_vocab.stoi, user_vocab, label_vocab)
            except:
                i += 1
                error_num += 1
                print('error:', error_num)
                print(line)
                continue
            result.append(res)
            if i % 10000 == 0:
                print(i)
            i += 1
    
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    return result

def build_test_sequence(output_file, doc_vocab, user_vocab):
    result = []
    with open(os.path.join(settings.TEST_DATA_DIR, 'testing_set_135089.txt'), 'r', encoding='UTF-8') as f:
        for line in f:
            res = get_test_indices(line, doc_vocab.stoi, user_vocab)
            result.append(res)
    print('test dataset len:', len(result))
    assert(len(result) == 135089)
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    return result

def load_user_vocab():
    with open(os.path.join(VOCAB_DIR, 'user_vocab.pk'), 'rb') as f:
        return pickle.load(f)

def load_label_vocab():
    with open(os.path.join(VOCAB_DIR, 'labels.pk'), 'rb') as f:
        return pickle.load(f)

def load_topic_vocab():
    with open(os.path.join(VOCAB_DIR, 'topic_vocab.pk'), 'rb') as f:
        return pickle.load(f)

def load_doc_vocab():
    with open(os.path.join(VOCAB_DIR, 'doc_vocab.pk'), 'rb') as f:
        return pickle.load(f)

def load_favs():
    with open(os.path.join(VOCAB_DIR, 'favs.pk'), 'rb') as f:
        return pickle.load(f)

def build_test_dataset():
    doc_vocab = load_doc_vocab()
    user_vocab = load_user_vocab()
    
    build_test_sequence(os.path.join(settings.TEST_DATA_DIR, 'test.pk'), doc_vocab, user_vocab)

def build_small_train_dataset():
    doc_vocab = load_doc_vocab()
    user_vocab = load_user_vocab()
    label_vocab = load_label_vocab()

    min_index = 0
    max_index = 1000
    for i in range(3):
        print('train{}.pk'.format(i))
        build_train_sequence_old(os.path.join(settings.VAL_DIR, 'val{}.pk'.format(i)), doc_vocab, user_vocab, label_vocab, min_index=min_index, max_index=max_index)
        min_index += 1000
        max_index += 1000

def build_vocabs():
    build_doc_vocab(os.path.join(VOCAB_DIR, 'doc_vocab.pk'))
    build_user_vocab(os.path.join(VOCAB_DIR, 'user_vocab.pk'))
    topic_vocab = build_topic_vocab(os.path.join(VOCAB_DIR, 'topic_vocab.pk'))
    build_favoriates(os.path.join(VOCAB_DIR, 'favs.pk'), topic_vocab)
    build_label_vocab(os.path.join(VOCAB_DIR, 'labels.pk'))

if __name__ == '__main__':

    # build vocab run once
    

    # Then build train files
    doc_vocab = load_doc_vocab()
    user_vocab = load_user_vocab()
    label_vocab = load_label_vocab()
    build_train_sequence(settings.TRAIN_DIR, doc_vocab, user_vocab, label_vocab)

    #min_index = 0
    #max_index = 1000000
    #for i in [1,2,10,11,12,13,14,15]:
    #    min_index = (i-1)*1000000
    #    max_index = min_index + 1000000 -1
    #    print('train{}.pk'.format(i))
    #    build_train_sequence_old(os.path.join(settings.TRAIN_DIR, 'train{}.pk'.format(i)), doc_vocab, user_vocab, label_vocab, min_index=min_index, max_index=max_index)


    #build_small_train_dataset()

    #vocab = build_doc_vocab(os.path.join(DATA_BIN_DIR, 'doc_vocab.pk'))
    #long2short = build_longid_dict()
    #print(len(vocab))
    #print(vocab.itos[:100])
    #print(get_read_indices(random_pick_train_lines(1)[0], vocab.stoi, long2short, user_vocab))

    #user_vocab = build_user_vocab(os.path.join(DATA_BIN_DIR, 'user_vocab.pk'))
    #with open(os.path.join(DATA_BIN_DIR, 'user_vocab.pk'), 'rb') as f:
    #    user_vocab = pickle.load(f)
    #    print(user_vocab.stoi)

    #build_favoriates(None, vocab, long2short)
    #build_topic_vocab(os.path.join(DATA_BIN_DIR, 'topic_vocab.pk'))
    #topic_vocab = load_topic_vocab()
    #build_favoriates(os.path.join(DATA_BIN_DIR, 'favs.pk'), topic_vocab)
    #label_vocab = build_label_vocab(os.path.join(DATA_BIN_DIR, 'labels.pk'))
    #print(label_vocab.itos[0])
    #print(len(label_vocab.itos))

    
    #topic_vocab = build_topic_vocab(os.path.join(DATA_BIN_DIR, 'topic_vocab.pk'))
    #build_favoriates(os.path.join(DATA_BIN_DIR, 'favs.pk'), topic_vocab)

    #min_index = 0
    #max_index = 10000
    #for i in range(5):
    #    print('train{}.pk'.format(i))
    #    build_train_sequence(os.path.join(DATA_BIN_DIR, 'test/train{}.pk'.format(i)), doc_vocab, user_vocab, label_vocab, min_index=min_index, max_index=max_index)
    #    min_index += 10000
    #    max_index += 10000

    #build_train_sequence(os.path.join(DATA_BIN_DIR, 'val10k.pk'), doc_vocab, user_vocab, label_vocab, min_index=24000000, max_index=24100000)
    
