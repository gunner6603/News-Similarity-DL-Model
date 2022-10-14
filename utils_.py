import json
import pickle
import numpy as np
from collections import Counter
import pandas as pd


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def make_json_list(path):
    with open(path, 'r') as f:
        json_list = list(f)
    return json_list


def make_vocab(list_of_json_lists, word_threshold, pos_list, vocab_path):
    
    counter = Counter()

    for json_list in list_of_json_lists:                
        for json_str in json_list:
            article = json.loads(json_str)
            sentences = article['sentences']
            for sentence in sentences:
                for token in sentence['tokens']:
                    if token['pos'][0] in pos_list:
                        counter[token['lemma']+'/'+token['pos']] += 1
                    
    words = [word for word, cnt in counter.items() if cnt >= word_threshold]
    
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')
    for word in words:
        vocab.add_word(word)
        
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    return vocab


def get_tf_and_idf(json_list1, json_list2, vocab, first_n=None):
    df = np.zeros(len(vocab))
    tfs_list = []
    for json_list in [json_list1, json_list2]:
        tfs = []
        for json_str in json_list:
            tf = np.zeros(len(vocab))
            article = json.loads(json_str)
            sentences = article['sentences']

            for i, sentence in enumerate(sentences):
                if first_n and i == first_n:
                    break
                for token in sentence['tokens']:
                    tf[vocab(token['lemma']+'/'+token['pos'])] += 1

            for i in range(len(vocab)):
                if tf[i]:
                    df[i] += 1

            tfs.append(tf)
        tfs_list.append(np.array(tfs))
        
    idf = np.log(tfs_list[0].shape[0]*len(tfs_list)/(1+df))
    
    return tfs_list, idf, df


def get_nnp_tf(json_list1, json_list2, vocab, first_n=None):
    tfs_list = []
    for json_list in [json_list1, json_list2]:
        tfs = []
        for json_str in json_list:
            tf = np.zeros(len(vocab))
            article = json.loads(json_str)
            sentences = article['sentences']

            for i, sentence in enumerate(sentences):
                if first_n and i == first_n:
                    break
                for token in sentence['tokens']:
                    if token['pos'] == 'NNP':
                        tf[vocab(token['lemma']+'/'+token['pos'])] += 1

            tfs.append(tf)
        tfs_list.append(np.array(tfs))
    
    return tfs_list

'''
def counter_cs(c1, c2):
    c_union = list(set(c1).union(set(c2)))
    n = len(c_union)
    tf1 = np.zeros(n)
    tf2 = np.zeros(n)
    for i in range(n):
        tf1[i] = c1[c_union[i]]
        tf2[i] = c2[c_union[i]]
    
    ip = np.sum(tf1*tf2)
    l2_1 = np.sqrt(np.sum(tf1**2))
    l2_2 = np.sqrt(np.sum(tf2**2))
    
    return ip / (l2_1*l2_2)


def get_nnp_cs(json_list1, json_list2):
    pair_num = len(json_list1)
    cs = np.zeros(pair_num)
    
    for idx in range(pair_num):
        
        json_str = json_list1[idx]
        article = json.loads(json_str)
        sentences = article['sentences']
        counter1 = Counter()
        for sentence in sentences:
            for token in sentence['tokens']:
                if token['pos'] == 'NNP':
                    counter1[token['lemma']+'/'+token['pos']] += 1

        json_str = json_list2[idx]
        article = json.loads(json_str)
        sentences = article['sentences']
        counter2 = Counter()
        for sentence in sentences:
            for token in sentence['tokens']:
                if token['pos'] == 'NNP':
                    counter2[token['lemma']+'/'+token['pos']] += 1
                    
        cs[idx] = counter_cs(counter1, counter2)
        
        return cs
'''

def get_tf_idf(tf, idf):
    return tf*idf


def get_cosine_similarity(m1, m2, eps=1e-10):
    
    ip = np.sum(m1*m2, axis=1)
    l2_1 = np.sqrt(np.sum(m1**2, axis=1))
    l2_2 = np.sqrt(np.sum(m2**2, axis=1))
    cos_sim = ip/(l2_1*l2_2+eps)
    
    return cos_sim


def get_labels(path):
    df_ = pd.read_table(path, header=None)
    labels = [1 if label=='O' else 0 for label in df_[0]]
    return np.array(labels)


def get_score_threshold_and_print_accuracy(scores, labels):
        
    optimal_score_threshold = 0
    acc = 0
    for i in range(20000):
        tmp_score_threshold = np.random.uniform(0,1)
        correct = np.sum((scores > tmp_score_threshold) == labels)
        tmp_acc = correct / len(labels)
        if tmp_acc > acc:
            acc = tmp_acc
            optimal_score_threshold = tmp_score_threshold
            
    print(f'accuracy : {acc:0.4f}')
    
    return optimal_score_threshold, acc


def print_test_statistics(predictions, labels):
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(len(labels)):
        prediction = predictions[i]
        label = labels[i]
        if prediction == 1:
            if label == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label == 1:
                FN += 1
            else:
                TN += 1
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall / (precision + recall)
    
    print(f'accuracy : {acc:0.4f}')
    print(f'precision : {precision:0.4f}')
    print(f'recall : {recall:0.4f}')
    print(f'F1 score : {f1_score:0.4f}')
    print(f'TP : {TP:4d}  FN : {FN:4d}')
    print(f'FP : {FP:4d}  TN : {TN:4d}')


def preprocess(json_list, vocab):
    document_list = []
    for json_str in json_list:
        article = json.loads(json_str)
        sentence_list = []
        for sentence in article['sentences']:
            token_list = []
            for token in sentence['tokens']:
                vocab_idx = vocab(token['lemma']+'/'+token['pos'])
                if vocab_idx != vocab('<unk>'):
                    token_list.append(vocab_idx)
            if len(token_list) >= 2:
                sentence_list.append(token_list)
        document_list.append(sentence_list)
    return document_list