# -*- coding utf-8 -*-
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))

import nltk
import torch
import torch.utils.data as torchdata
from common.vocabulary import Vocab


class CustomDataset(torchdata.Dataset):
    def __init__(self, datas, window=2, unk_tkn='<unk>', pad_tkn='<pad>', device=None):
        self.window = window
        self.device = device
        self.unk_tkn = unk_tkn
        self.pad_tkn = pad_tkn
        self.flatten = lambda d: [tkn for sent in d for tkn in sent]
        # build_vocabulary
        self.vocab = Vocab(unk_tkn=self.unk_tkn, pad_tkn=self.pad_tkn)
        self.vocab.build_vocab(datas)
        self.get_idx = lambda x: self.vocab.stoi.get(x) if self.vocab.stoi.get(x) else self.vocab.stoi.get(self.unk_tkn)
        # build data
        self.data = self._build_data(datas)

    def _build_data(self, datas):
        all_corpus = self.flatten(list(nltk.ngrams(self._add_pad(s), 2 * self.window + 1)) for s in datas)
        train_data = []
        for tkns in all_corpus:
            for i in range(self.window * 2 + 1):
                if tkns[i] == self.pad_tkn or i == self.window:
                    continue
                train_data.append((tkns[self.window], tkns[i]))
        return train_data

    def _add_pad(self, s):
        assert isinstance(s, list)
        s = [self.pad_tkn] * self.window + s + [self.pad_tkn] * self.window
        return s

    def _numerical(self, s):
        return list(map(self.get_idx, s))

    def collate_fn(self, data):
        """
        need a custom 'collate_fn' function in 'torchdata.DataLoader' for variable length of dataset
        """
        x = []
        y = []
        for d in data:
            d = self._numerical(d)
            x.append(d[0])
            y.append(d[1])
        if self.device == 'cuda':
            return torch.LongTensor(x).view(-1, 1).cuda(), torch.LongTensor(y).view(-1, 1).cuda()
        else:
            return torch.LongTensor(x).view(-1, 1), torch.LongTensor(y).view(-1, 1)

    def __getitem__(self, index):
        # return index datas
        return self.data[index]

    def __len__(self):
        # lengths of data
        return len(self.data)
