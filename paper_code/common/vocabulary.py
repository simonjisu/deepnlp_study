# -*- coding utf-8 -*-
from collections import defaultdict

class Vocab(object):
    def __init__(self, unk_tkn='<unk>', pad_tkn='<pad>'):
        self.stoi = defaultdict()
        self.itos = None
        self.unk_tkn = unk_tkn
        self.pad_tkn = pad_tkn
    
    def build_vocab(self, all_tokens, unk_idx=0, pad_idx=1):
        """
        all_tokens ndim = 2
        """
        # Build Vocabulary
        flatten = lambda x: [tkn for sen in x for tkn in sen]
        total_words = flatten(all_tokens)
        
        self.stoi[self.unk_tkn] = unk_idx
        self.stoi[self.pad_tkn] = pad_idx
        for token in total_words:
            if not self.stoi.get(token):
                self.stoi[token] = len(self.stoi)
        self.itos = [t for t, i in sorted([(token, index) for token, index in self.stoi.items()], key=lambda x: x[1])]
           
    def __len__(self):
        return len(self.stoi)