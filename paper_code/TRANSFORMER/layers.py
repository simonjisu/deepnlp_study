import torch
import torch.nn as nn
from sublayers import MultiHeadAttention, PositionWiseFFN


# Encode Layer
class Encode_Layer(nn.Module):
    """encode layer"""
    def __init__(self, n_head, d_model, d_k, d_v, d_f, drop_rate=0.1, use_conv=False):
        super(Encode_Layer, self).__init__()
        self.selfattn = MultiHeadAttention(n_head, d_model, d_k, d_v, drop_rate=drop_rate)
        self.pwffn = PositionWiseFFN(d_model, d_f, drop_rate=drop_rate, use_conv=use_conv)
        self.norm_selfattn = nn.LayerNorm(d_model)
        self.norm_pwffn = nn.LayerNorm(d_model)
        
    def forward(self, enc_input, enc_mask=None):
        """
        Inputs:
        * enc_input: (B, T, d_model)
        * enc_mask: (B, T, T)
        -------------------------------------
        Outputs:
        * enc_output: (B, T, d_model)
        * enc_attn: (n_head * B, T, T)
        """
        # Layer: Multi-Head Attention + Add & Norm
        # encode self-attention
        enc_output, enc_attn = self.selfattn(enc_input, enc_input, enc_input, mask=enc_mask)
        enc_output = self.norm_selfattn(enc_input + enc_output)
        
        # Layer: PositionWiseFFN + Add & Norm
        pw_output = self.pwffn(enc_output)
        enc_output = self.norm_pwffn(enc_output + pw_output)
        
        return enc_output, enc_attn

# Decode Layer
class Decode_Layer(nn.Module):
    """decode layer"""
    def __init__(self, n_head, d_model, d_k, d_v, d_f, drop_rate=0.1, use_conv=False):
        super(Decode_Layer, self).__init__()
        self.selfattn_masked = MultiHeadAttention(n_head, d_model, d_k, d_v, drop_rate=drop_rate)
        self.dec_enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, drop_rate=drop_rate)
        self.pwffn = PositionWiseFFN(d_model, d_f, drop_rate=drop_rate, use_conv=use_conv)
        self.norm_selfattn_masked = nn.LayerNorm(d_model)
        self.norm_dec_enc_attn = nn.LayerNorm(d_model)
        self.norm_pwffn = nn.LayerNorm(d_model)
    
    def forward(self, dec_input, enc_output, dec_self_mask=None, dec_enc_mask=None):
        """
        Inputs:
        * dec_input: (B, T_q, d_model)
        * enc_input: (B, T, d_model)
        * dec_self_mask: (B, T_q, T_q)
        * dec_enc_mask: (B, T_q, T)
        -------------------------------------
        Outputs:
        * dec_output: (B, T_q, d_model)
        * dec_self_attn: (n_head * B, T_q, T_q)
        * dec_enc_attn: (n_head * B, T_q, T)
        """
        # Layer: Multi-Head Attention + Add & Norm
        # decode self-attention
        dec_self_output, dec_self_attn = self.selfattn_masked(dec_input, dec_input, dec_input, 
                                                              mask=dec_self_mask)
        dec_self_output = self.norm_selfattn_masked(dec_input + dec_self_output)
        
        # Layer: Multi-Head Attention + Add & Norm
        # decode output(queries) + encode output(keys, values)
        dec_output, dec_enc_attn = self.dec_enc_attn(dec_self_output, enc_output, enc_output, 
                                                     mask=dec_enc_mask)
        dec_output = self.norm_dec_enc_attn(dec_self_output + dec_output)
        
        # Layer: PositionWiseFFN + Add & Norm
        pw_output = self.pwffn(dec_output)
        dec_output = self.norm_pwffn(dec_output + pw_output)
        
        return dec_output, dec_self_attn, dec_enc_attn