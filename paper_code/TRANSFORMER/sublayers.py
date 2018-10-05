# -*- coding utf-8 -*- 
# author: https://github.com/simonjisu

import torch
import torch.nn as nn
import numpy as np
from modules import XavierLinear, ScaledDotProductAttention

# Multi-head Attention
class MultiHeadAttention(nn.Module):
    """Multi-head Attention"""
    def __init__(self, n_head, d_model, d_k, d_v, drop_rate=0.1, return_attn=True):
        """
        paper setting: n_head = 8, d_k = d_v = d_model / n_head = 64
        Multi-head attention allows the model to jointly attend to information from 
        different representation subspaces at different positions.
        with a single attention head, averaging inhibits this.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.return_attn = return_attn
        self.linear_q = XavierLinear(d_model, n_head*d_k)
        self.linear_k = XavierLinear(d_model, n_head*d_k)
        self.linear_v = XavierLinear(d_model, n_head*d_v)
        self.linear_o = XavierLinear(n_head*d_v, d_model)
        self.attention = ScaledDotProductAttention(d_k, return_attn=return_attn)
        self.drop_out = nn.Dropout(drop_rate)
        
    def forward(self, q, k, v, mask=None):
        """
        Inputs:
        * q: (B, T_q, d_model)
        * k: (B, T_k, d_model)
        * v: (B, T_v, d_model)
        * mask: (B, T_q, T_k)
        ---------------------
        Outputs:
        * output: (B, T_q, d_model)
        * attn: (n_head * B, T_q, T_k)
        """
        n_head, d_model, d_k, d_v = self.n_head, self.d_model, self.d_k, self.d_v
        B, T_q, _ = q.size()
        B, T_k, _ = k.size()
        B, T_v, _ = v.size()
        # through linear layer: 
        # lin_qs : (B, T_q, d_model) --> (B, T_q, n_head * d_k) --> (n_head * B, T_q, d_k)
        # lin_ks : (B, T_k, d_model) --> (B, T_k, n_head * d_k) --> (n_head * B, T_k, d_k) 
        # lin_vs : (B, T_v, d_model) --> (B, T_v, n_head * d_v) --> (n_head * B, T_v, d_v)
        lin_qs = self.linear_q(q).view(B, T_q, n_head, d_k)  
        lin_ks = self.linear_k(k).view(B, T_k, n_head, d_k)  
        lin_vs = self.linear_v(v).view(B, T_v, n_head, d_v)  
        lin_qs = lin_qs.permute(2, 0, 1, 3).contiguous().view(-1, T_q, d_k)
        lin_ks = lin_ks.permute(2, 0, 1, 3).contiguous().view(-1, T_k, d_k)
        lin_vs = lin_vs.permute(2, 0, 1, 3).contiguous().view(-1, T_v, d_v)
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        # attention: Scaled Dot-Product Attention
        ## heads: (n_head * B, T_q, d_v)
        ## attn: (n_head * B, T_q, T_k)
        if self.return_attn:
            heads, attn = self.attention(q=lin_qs, k=lin_ks, v=lin_vs, mask=mask)
        else:
            heads = self.attention(q=lin_qs, k=lin_ks, v=lin_vs, mask=mask)
        # concat
        heads_cat = torch.cat(list(heads.chunk(n_head, dim=0)), dim=-1)  # (n_head * B, T_q, d_v) --> (B, T_q, n_head * d_v)
        output = self.linear_o(heads_cat)  # (B, T_q, n_head * d_v) --> (B, T_q, d_model)
        output = self.drop_out(output)
        if self.return_attn:
            return output, attn
        return output
    

# Position-wise Feed-Forward Networks
class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Networks"""
    def __init__(self, d_model, d_f, drop_rate=0.1, use_conv=False):
        super(PositionWiseFFN, self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.fc = nn.Sequential(
                nn.Conv1d(d_model, d_f, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(d_f, d_model, kernel_size=1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_f),
                nn.ReLU(),
                nn.Linear(d_f, d_model)
            )
        self.drop_out = nn.Dropout(drop_rate)
    
    def forward(self, x):
        """
        Inputs:
        x: (B, T, d_model)
        -----------------------
        Ouputs:
        output: (B, T, d_model)
        """
        if self.use_conv:
            x = x.transpose(1, 2)  # (B, T, d_model) --> (B, d_model, T), reshape like (batch, channel, dim)
            output = self.fc(x).transpose(1, 2)  # (B, d_model, T) --> (B, T, d_model)
        else:
            output = self.fc(x)
            
        output = self.drop_out(output)
        return output
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, n_pos, d_model, pad_idx=0):
        """
        n_pos = max sequence length + 1
        """
        super(PositionalEncoding, self).__init__()
        self.n_pos = n_pos
        self.d_model = d_model
        self.pe_table = np.array(self.get_pe_table())
        self.pe_table[:, 0::2] = np.sin(self.pe_table[:, 0::2])
        self.pe_table[:, 1::2] = np.cos(self.pe_table[:, 1::2])
        self.pe_table[pad_idx, :] = 0  # embed all pad to 0
        self.pe = nn.Embedding.from_pretrained(torch.FloatTensor(self.pe_table), freeze=True)
        
    def cal_angle(self, pos, hid_idx):
        return pos / (10000 ** ((2*(hid_idx // 2) / self.d_model)) )
    
    def get_pe_table(self):
        return [[self.cal_angle(pos, i) for i in range(self.d_model)] for pos in range(self.n_pos)]         
        
    def forward(self, inputs):
        return self.pe(inputs)
    
# Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_len, d_model, pad_idx=0):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_len, d_model, padding_idx=pad_idx)
        
    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.d_model)
