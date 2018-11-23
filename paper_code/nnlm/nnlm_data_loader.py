import os
import re
import json
import unicodedata
import torch
import torch.utils.data as torchdata
from collections import defaultdict
from nltk.util import ngrams
from common.vocabulary import Vocab

# Dataset
class DataSet(object):
    def __init__(self, base_path='../data/', train=None, valid=None, test=None, n_gram=3, tokenizer=None, 
                 save_tokens=False, direct_load=False, remove_short=True, fixed_vocab=None, device=None):
        self.base_path = base_path
        self.train = train
        self.valid = valid
        self.test = test
        self.n_gram = n_gram
        self.save = save_tokens
        self.direct_load = direct_load
        self._remove_short = remove_short
        self.device = device
#         self.fixed_vocab = fixed_vocab
        self.tokenizer = str.split if self.direct_load else tokenizer
        self.path_dict = dict((name, getattr(self, name)) for name in ['train', 'valid', 'test']
                              if getattr(self, name) is not None)
    def splits(self):
        datasets = []
        for k, path in self.path_dict.items():
            if k == 'train':
                data = self.load_and_tokenization(path_=os.path.join(self.base_path, path))
                if self.save:
                    self.save_tokens(os.path.join(self.base_path, k+'_tokens'), data)                    
                dataset = CustomDataset(data, 
                                        n_gram=self.n_gram, 
                                        train=True, 
                                        remove_short=self._remove_short,
                                        device=self.device,
                                        fixed_vocab=self.fixed_vocab)
                datasets.append(dataset)
            else:
                data = self.load_and_tokenization(path_=os.path.join(self.base_path, path))
                if self.save:
                    self.save_tokens(os.path.join(self.base_path, k+'_tokens'), data)
                dataset = CustomDataset(data, 
                                        n_gram=self.n_gram, 
                                        train=False, 
                                        remove_short=self._remove_short, 
                                        vocabulary=datasets[0].vocab,
                                        device=self.device)
                datasets.append(dataset)
        
        return datasets

    def _unicode_to_ascii(self, s):
        return ''.join( c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn' )
    
    def _preprocessing(self, s):
        s = self._unicode_to_ascii(s)
        s = re.sub(r"([^가-힣A-Za-z0-9\s])", r" \1 ", s)
        s = re.sub(r"([^가-힣A-Za-z0-9\s])(?=.*\1)", r"", s)
        return s.strip().lower()
    
    def load_and_tokenization(self, path_):
        # Load data
        with open(path_, 'r') as file:
            data = file.read().splitlines()
        if self.direct_load:
            assert self.tokenizer == str.split, 'tokenizer must be "str.split", when using direct_load'
            return [self.tokenizer(s) for s in data]
        else:
            data = [self._preprocessing(s) for s in data]
            data = [self.tokenizer(s) for s in data]  # Tokenization
            return data
    
    def save_tokens(self, path_, data):
        with open(path_, 'w', encoding='utf-8') as file:
            for d in data:
                print(' '.join(d), file=file)
    
    def create_loader(self, train=None, valid=None, test=None, batch_size=32, shuffle=False, drop_last=False):
        temp_dict = {'train': train, 'valid': valid, 'test': test}
        result = []
        for x in self.path_dict.keys():
            temp = torchdata.DataLoader(dataset=temp_dict[x], 
                                        collate_fn=temp_dict[x].collate_fn,
                                        batch_size=batch_size, 
                                        shuffle=shuffle, 
                                        drop_last=drop_last)
            result.append(temp)
        return result

# Vocab
# class Vocab(object):
#     def __init__(self):
#         self.stoi = defaultdict()
#         self.itos = None
        
#     def build_vocab(self, all_tokens, fixed_vocab=None):
#         """
#         fixed vocab must be a path
#         """
#         if fixed_vocab:
#             self.load(path=fixed_vocab)
#         else:
#             # Build Vocabulary
#             flatten = lambda x: [tkn for sen in x for tkn in sen]
#             unique_tokens = set(flatten(all_tokens))
#             self.stoi['<unk>'] = 0
#             self.stoi['<pad>'] = 1
#             for i, token in enumerate(unique_tokens, 2):
#                 self.stoi[token] = i
#         self.itos = [t for t, i in sorted(
#             [(token, index) for token, index in self.stoi.items()], key=lambda x: x[1])]
            
#     def save(self, path, vocab_dict):
#         js = json.dumps(vocab_dict)
#         with open(path, 'w', encoding='utf-8') as file:
#             print(js, file=file)
            
#     def load(self, path):
#         with open(path, 'r', encoding='utf-8') as file:
#             js = file.read()
#         self.stoi = json.loads(js)
    
    def __len__(self):
        return len(self.stoi)
    
# CustomDataset
class CustomDataset(torchdata.Dataset):
    def __init__(self, data, n_gram=3, train=True, remove_short=True, vocabulary=None, device=None):
        self.n_gram = n_gram
        self.func_ngrams = ngrams
        self._remove_short = remove_short
        self._removed = 0
        self._total = 0
        self.device = device
        if train:
            self.vocab = Vocab()
            self.data = self.create_dataset(data, train=train)
        else:
            self.vocab = vocabulary
            self.data = self.create_dataset(data)

    def create_dataset(self, data, train=False):
        if self._remove_short:
            data_ = []
            for i, s in enumerate(data):
                self._total += 1
                if len(s) >= self.n_gram:
                    data_.append(s)
                else:
                    self._removed += 1
        else:
            data_ = data
        if train:
            self.vocab.build_vocab(data_)
                
        data_ = self.numerical(data_)
        data_ = self.get_ngrams(data_)
        return data_    

    def numerical(self, all_tokens):
        # Numericalize all tokens
        f = lambda x: self.vocab.stoi.get(x) if x in self.vocab.stoi.keys() else self.vocab.stoi['<unk>']
        all_tokens_numerical = [list(map(f, s)) for s in all_tokens]
        return all_tokens_numerical            
    
    def get_ngrams(self, all_tokens_numerical):
        # create n-grams data
        data = []
        x = []
        y = []
        for sent in all_tokens_numerical:
#             if len(sent) < self.n_gram:
#                 sent.append(self.vocab.stoi['<pad>']*(self.n_gram - len(sent)))
            for grams in list(self.func_ngrams(sent, self.n_gram)):
                data.append(list(grams))
        return data
    
    def collate_fn(self, data):
        """
        need a custom 'collate_fn' function in 'torchdata.DataLoader' for variable length of dataset
        """
        x = []
        y = []
        for d in data:
            x.append(d[:(self.n_gram-1)])
            y.extend(d[(self.n_gram-1):])
        if self.device == 'cuda':
            return torch.LongTensor(x).cuda(), torch.LongTensor(y).cuda()
        else:
            return torch.LongTensor(x), torch.LongTensor(y)
    
    def __getitem__(self, index):
        # return index datas
        return self.data[index]

    def __len__(self):
        # lengths of data
        return len(self.data)