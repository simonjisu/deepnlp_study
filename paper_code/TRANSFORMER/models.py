# -*- coding utf-8 -*- 
# author: https://github.com/simonjisu

import torch
import torch.nn as nn

from layers import Encode_Layer, Decode_Layer
from sublayers import PositionalEncoding, Embedding
from modules import XavierLinear
from utils import get_padding_mask

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_len, max_seq_len, n_layer, n_head, d_model, d_k, d_v, d_f, 
                 drop_rate=0.1, use_conv=False, return_attn=True, pad_idx=0):
        super(Encoder, self).__init__()
        self.pad_idx = pad_idx
        self.return_attn = return_attn
        self.dropout = nn.Dropout(drop_rate)
        self.embed_layer = Embedding(vocab_len, d_model, pad_idx=pad_idx)
        self.pos_layer = PositionalEncoding(max_seq_len+1, d_model)
        self.layers = nn.ModuleList([Encode_Layer(n_head, d_model, d_k, d_v, d_f, 
                                                  drop_rate=drop_rate, 
                                                  use_conv=use_conv,
                                                  return_attn=return_attn) \
                                     for i in range(n_layer)])
        
    def forward(self, enc, enc_pos):
        """
        Inputs:
        * enc: (B, T)
        * enc_pos: (B, T)
        -------------------------------------
        Outputs:
        * enc_output: (B, T, d_model)
        * self_attns: (n_head*B, T, T)
        """
        self_attns = []  # (n_layer, n_head*B, T, T)
        # self attention padding mask: (B, T, T)
        attn_mask = get_padding_mask(q=enc, k=enc, pad_idx=self.pad_idx, mode='attn')
        non_pad_mask = get_padding_mask(q=enc, pad_idx=self.pad_idx, mode='nonpad')
        # embedding + position encoding: (B, T) --> (B, T, d_model)
        enc_output = self.embed_layer(enc) + self.pos_layer(enc_pos)
        enc_output = self.dropout(enc_output)
        # forward encode layer
        for enc_layer in self.layers:
            if self.return_attn:
                enc_output, enc_self_attn = enc_layer(enc_input=enc_output, 
                                                      enc_mask=attn_mask, 
                                                      non_pad_mask=non_pad_mask)
                self_attns.append(enc_self_attn)
            else:
                enc_output = enc_layer(enc_input=enc_output, 
                                       enc_mask=attn_mask, 
                                       non_pad_mask=non_pad_mask)
        
        if self.return_attn:
            return enc_output, self_attns
        return enc_output
    
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_len, max_seq_len, n_layer, n_head, d_model, d_k, d_v, d_f, 
                 pad_idx=0, drop_rate=0.1, use_conv=False, return_attn=True):
        super(Decoder, self).__init__()
        self.pad_idx = pad_idx
        self.return_attn = return_attn
        self.dropout = nn.Dropout(drop_rate)
        self.embed_layer = Embedding(vocab_len, d_model, pad_idx=pad_idx)
        self.pos_layer = PositionalEncoding(max_seq_len+1, d_model)
        self.layers = nn.ModuleList([Decode_Layer(n_head, d_model, d_k, d_v, d_f, 
                                                  drop_rate=drop_rate, 
                                                  use_conv=use_conv,
                                                  return_attn=return_attn) \
                                     for i in range(n_layer)])
        
    def forward(self, dec, dec_pos, enc, enc_output):
        """
        Inputs:
        * dec: (B, T_q)
        * dec_pos: (B, T_q)
        * enc: (B, T)
        * enc_output: (B, T, d_model)
        -------------------------------------
        Outputs:
        * dec_output: (B, T_q, d_model)
        * self_attns: (n_head*B, T_q, T_q)
        * dec_enc_attns: (n_haed*B, T_q, T)
        """
        self_attns = []  # (n_layer, n_head*B, T_q, T_q)
        dec_enc_attns = []  # (n_layer, n_head*B, T_q, T)
        
        # self attention padding mask: (B, T_q, T)
        non_pad_mask = get_padding_mask(q=dec, pad_idx=self.pad_idx, mode='nonpad')
        attn_mask = get_padding_mask(q=dec, k=dec, pad_idx=self.pad_idx, mode='attn')
        subseq_mask = get_padding_mask(q=dec, mode='subseq')
        self_attn_mask = (attn_mask + subseq_mask).gt(0)
        # enc_dec attention padding mask
        non_pad_mask = get_padding_mask(q=dec, pad_idx=self.pad_idx, mode='nonpad')
        dec_enc_attn_mask = get_padding_mask(q=dec, k=enc, pad_idx=self.pad_idx, mode='attn')
        
        # embedding + position encoding: (B, T) --> (B, T, d_model)
        dec_output = self.embed_layer(dec) + self.pos_layer(dec_pos)
        dec_output = self.dropout(dec_output)
        # forward decode layer
        for dec_layer in self.layers:
            if self.return_attn:
                dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_input=dec_output, 
                                                                    enc_output=enc_output, 
                                                                    dec_self_mask=self_attn_mask, 
                                                                    dec_enc_mask=dec_enc_attn_mask,
                                                                    non_pad_mask=non_pad_mask)
                self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)
            else:
                dec_output = dec_layer(dec_input=dec_output, 
                                       enc_output=enc_output, 
                                       dec_self_mask=self_attn_mask, 
                                       dec_enc_mask=dec_enc_attn_mask,
                                       non_pad_mask=non_pad_mask)
        
        if self.return_attn:
            return dec_output, self_attns, dec_enc_attns
        return dec_output

    
# Transformer Model
class Transformer(nn.Module):
    def __init__(self, enc_vocab_len, enc_max_seq_len, dec_vocab_len, dec_max_seq_len, 
                 n_layer, n_head, d_model, d_k, d_v, d_f, 
                 pad_idx=0, drop_rate=0.1, use_conv=False, return_attn=True,
                 linear_weight_share=True, embed_weight_share=True):
        super(Transformer, self).__init__()
        self.return_attn = return_attn
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        self.encoder = Encoder(enc_vocab_len, enc_max_seq_len, n_layer, n_head, 
                               d_model, d_k, d_v, d_f, 
                               pad_idx=pad_idx, 
                               drop_rate=drop_rate, 
                               use_conv=use_conv, 
                               return_attn=return_attn)
        self.decoder = Decoder(dec_vocab_len, dec_max_seq_len, n_layer, n_head, 
                               d_model, d_k, d_v, d_f,
                               pad_idx=pad_idx, 
                               drop_rate=drop_rate, 
                               use_conv=use_conv, 
                               return_attn=return_attn)
        self.projection = XavierLinear(d_model, dec_vocab_len)
        if linear_weight_share:
            # share the same weight matrix between the decoder embedding layer 
            # and the pre-softmax linear transformation
            self.projection.linear.weight = self.decoder.embed_layer.embedding.weight
        
        if embed_weight_share:
            # share the same weight matrix between the decoder embedding layer 
            # and the encoder embedding layer
            assert enc_vocab_len == dec_vocab_len, "vocab length must be same"
            self.encoder.embed_layer.embedding.weight = self.decoder.embed_layer.embedding.weight
            
    def forward(self, enc, enc_pos, dec, dec_pos):
        """
        Inputs:
        * enc: (B, T)
        * enc_pos: (B, T)
        * dec: (B, T_q)
        * dec_pos: (B, T_q)
        -------------------------------------
        Outputs:
        * dec_output: (B*T_q, d_model)
        * attns_dict:
            * enc_self_attns: (n_head*B, T, T)
            * dec_self_attns: (n_head*B, T_q, T_q)
            * dec_enc_attns: (n_haed*B, T_q, T)
        """
        if self.return_attn:
            enc_output, enc_self_attns = self.encoder(enc, enc_pos)
            dec_output, dec_self_attns, dec_enc_attns = self.decoder(dec, dec_pos, enc, enc_output)
            dec_output = self.projection(dec_output)
            attns_dict = {'enc_self_attns': enc_self_attns, 
                         'dec_self_attns': dec_self_attns,
                         'dec_enc_attns': dec_enc_attns}
            return dec_output.view(-1, dec_output.size(2)), attns_dict
        else:
            enc_output = self.encoder(enc, enc_pos)
            dec_output = self.decoder(dec, dec_pos, enc, enc_output)
            dec_output = self.projection(dec_output)
            return dec_output.view(-1, dec_output.size(2))