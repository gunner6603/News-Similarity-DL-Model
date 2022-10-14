import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class NewsDataset(Dataset):
    def __init__(self, doc_list1, doc_list2, labels, vocab, max_doc_len, max_sent_len, first_n_words=None):
        self.doc_list1 = doc_list1
        self.doc_list2 = doc_list2
        self.labels = labels
        self.vocab = vocab
        self.max_doc_len = max_doc_len
        self.max_sent_len = max_sent_len
        self.first_n_words = first_n_words
                
    def __getitem__(self, index):
        items = []

        if self.first_n_words:
            for doc_list in [self.doc_list1, self.doc_list2]:
                cnt = 0
                n_words = []
                doc = doc_list[index]
                for sentence in doc:
                    for token in sentence:
                        n_words.append(token)
                        cnt += 1
                        if cnt == self.first_n_words:
                            break
                    if cnt == self.first_n_words:
                        break
                if cnt < self.first_n_words:
                    n_words += [self.vocab('<pad>')]*(self.first_n_words-cnt)
                items.append(torch.tensor(n_words, dtype=torch.int64))
            label = self.labels[index]

            return items[0], items[1], label

        for doc_list in [self.doc_list1, self.doc_list2]:
            item = torch.full((self.max_doc_len, self.max_sent_len), self.vocab('<pad>'), dtype=torch.int64)
            doc = doc_list[index]
            for i, sentence in enumerate(doc):
                if i < self.max_doc_len:
                    for j, token in enumerate(sentence):
                        if j < self.max_sent_len:
                            # item[self.max_doc_len-1-i, self.max_sent_len-1-j] = token
                            item[i, j] = token
                        else:
                            pass
                else:
                    pass
            items.append(item)
        label = self.labels[index]

        return items[0], items[1], label

    def __len__(self):
        return len(self.doc_list1)