import torch
import torch.utils.data as torchdata
import numpy as np
import unicodedata
import re
from vocabulary import Vocab

class TranslateDataset(torchdata.Dataset):
    def __init__(self, path='../data/translation/de_en_small.txt', exts='src-trg', vocab=None,
                 unk='<unk>', pad='<pad>', sos=None, eos=None, device=None, return_len=False,
                 cons=10e7):
        """
        if valid: vocab must be a list of tuples like [('src', src_vocab), ('trg', trg_vocab)] 
        return (src, srt_pos), (trg, trg_pos)
        """
        self.device = device
        self.unk_tkn = unk
        self.pad_tkn = pad
        self.sos_tkn = sos
        self.eos_tkn = eos
        self.cons = cons 
        self.vocab = vocab
#         self.return_len = return_len
        
        self.flatten = lambda d: [tkn for sent in d for tkn in sent]
        # call data
        data_dict = self._split_data(path)
        # build_vocabulary & datas
        self._build_datas(exts, data_dict)
        # TODO: return length of data and sort them
    
    def _split_data(self, path):
        with open(path, 'r') as file:
            data = file.read().splitlines()
        data = [line.split('\t') for line in data]
        data = [[self._normalize_string(src), self._normalize_string(trg)] \
                for (src, trg) in data]
        src_data, trg_data = list(zip(*data))
        src_data, trg_data = [x.split() for x in src_data], [x.split() for x in trg_data]
        if self.sos_tkn:
            src_data = [[self.sos_tkn] + x for x in src_data]
            trg_data = [[self.sos_tkn] + x for x in trg_data]
        if self.eos_tkn:
            src_data = [x + [self.eos_tkn] for x in src_data]
            trg_data = [x + [self.eos_tkn] for x in trg_data]
        self.src_maxlen = self._get_length(src_data)
        self.trg_maxlen = self._get_length(trg_data)
        data_dict = {
                    'src': src_data, 
                    'trg': trg_data
                    }
        
        return data_dict
    
    def _build_datas(self, exts, data_dict):
        assert (exts == 'src-trg') or (exts=='trg-src'), 'Error: exts must be either "src-trg" or "trg-src"'
        src, trg = exts.split('-')
        if self.vocab is None:   
            self.src_vocab = self._build_vocab(src, data_dict)
            self.trg_vocab = self._build_vocab(trg, data_dict)
        else:
            assert isinstance(self.vocab, list), \
                "if valid: vocab must be a list of tuples like [('src', src_vocab), ('trg', trg_vocab)] "
            self.src_vocab = dict(self.vocab)[src]
            self.trg_vocab = dict(self.vocab)[trg]
        self.data = [(s, t) for s, t in zip(data_dict[src], data_dict[trg])]
    
    def _build_vocab(self, typ, data_dict):
        vocab = Vocab(unk_tkn=self.unk_tkn, pad_tkn=self.pad_tkn, 
                      sos_tkn=self.sos_tkn, eos_tkn=self.eos_tkn)
        vocab.build_vocab(data_dict[typ])
        return vocab
    
    def _numerical(self, sent, vocab):
        get_idx = lambda x: vocab.stoi.get(x) if vocab.stoi.get(x) is not None else vocab.stoi.get(self.unk_tkn)
        return list(map(get_idx, sent))
            
    def _pad_data(self, sent, maxlen):
        assert isinstance(sent, list)
        if len(sent) < maxlen:
            return sent + [self.pad_tkn]*(maxlen-len(sent))
        else:
            return sent
    
    def _get_pos(self, x, pad_idx, cons=10e7):
        mask = np.equal(np.array(x), pad_idx)
        pos = np.ma.array(np.array(x), mask=mask, fill_value=cons).filled()
        pos = np.sort(pos).argsort() + 1
        pos = np.ma.array(pos, mask=mask, fill_value=0).filled()
        return pos
    
    def _get_length(self, data):
        return max([len(x) for x in data])

    def collate_fn(self, data):
        """
        need a custom 'collate_fn' function in 'torchdata.DataLoader' for variable length of dataset
        """
        src_data, trg_data = list(zip(*data))
        src = []
        trg = []
        batch_src_maxlen = self._get_length(src_data)
        batch_trg_maxlen = self._get_length(trg_data)
        
        for s, t in data:
            s = self._pad_data(s, batch_src_maxlen)
            s = self._numerical(s, vocab=self.src_vocab)
            src.append(s)
            t = self._pad_data(t, batch_trg_maxlen)
            t = self._numerical(t, vocab=self.trg_vocab)
            trg.append(t)
            
        src_pos = self._get_pos(src, pad_idx=self.src_vocab.stoi[self.pad_tkn], cons=self.cons)
        trg_pos = self._get_pos(trg, pad_idx=self.trg_vocab.stoi[self.pad_tkn], cons=self.cons)
        src, src_pos, trg, trg_pos = \
            map(lambda x: torch.LongTensor(x).to(self.device), [src, src_pos, trg, trg_pos])
#         TODO: return length of data and sort them
#         get_len = lambda x: sorted([len(seq) for seq in x], reverse=True)
#         if self.return_len:
#             return (src, trg)
        return src, src_pos, trg, trg_pos
    
    def _unicode_to_ascii(self, s):
        return ''.join( c for c in unicodedata.normalize('NFD', s) \
                       if unicodedata.category(c) != 'Mn' )

    def _normalize_string(self, s):
        s = self._unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([,.!?\"\'\-])", r" \1 ", s)
        s = re.sub(r"[^a-zA-Z0-9,.!?\"\'\-]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    
    def __getitem__(self, index):
        # return index datas
        return self.data[index]

    def __len__(self):
        # lengths of data
        return len(self.data)
