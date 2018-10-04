# -*- coding utf-8 -*- 
# author: https://github.com/simonjisu

import torch
import torch.nn as nn


def get_padding_mask(q, k=None, pad_idx=1, mode='attn'):
    """
    mode: attn
    > mask out for pad in attention with queries & keys sequences
    > return shape: (B, T_q, T_k)
    mode: subseq
    > mask out next tokens to preserve 'auto-regressive property'
    > return shape: (B, T_q, T_q)
    """
    B, q_len = q.size()
    if mode == 'attn':
        assert k is not None, "must have key sequences"
        padding_mask = k.eq(pad_idx)
        padding_mask = padding_mask.unsqueeze(1).expand(B, q_len, -1)
        return padding_mask
    elif mode == 'nonpad':
        # to mask out pad rows
        assert k is None, "don't need key sequences"
        return q.ne(pad_idx).type(torch.float).unsqueeze(-1)
    elif mode =='subseq':
        assert k is None, "don't need key sequences"
        subseq_mask = torch.triu(torch.ones((q_len, q_len), device=q.device, dtype=torch.uint8), 
                                 diagonal=1)
        subseq_mask = subseq_mask.unsqueeze(0).expand(B, -1, -1)
        return subseq_mask

def check_dotproduct_dist(d_k, sampling_size=1, seq_len=1, threshold=1e-10):
    """
    to check "https://arxiv.org/abs/1706.03762" Paper page 4, annotation 4
    -------------------------------
    To illustrate why the dot products get large, 
    assume that the components of q and k are independent random variables 
    with mean 0 and variance 1.
    Then their dot product has mean 0 and variance d_k
    """
    def cal_grad(attn):
        y = torch.softmax(attn, dim=2)
        return y * (1-y)
    
    q = nn.init.normal_(torch.rand((sampling_size, seq_len, d_k)), mean=0, std=1)
    k = nn.init.normal_(torch.rand((sampling_size, seq_len, d_k)), mean=0, std=1)
    attn = torch.bmm(q, k.transpose(1, 2))
    print('size of vector d_k is {}, sampling result, dot product distribution has \n - mean: {:.4f}, \n - var: {:.4f}'.\
          format(d_k, attn.mean().item(), attn.var().item()))
    grad = cal_grad(attn)
    print( "count of gradients that smaller than threshod({}) is {}, {:.4f}%".format(
        threshold, grad.le(threshold).sum(), grad.le(threshold).sum().item()/grad.view(-1).size(0)*100 ) )
    attn2 = attn / torch.sqrt(torch.as_tensor(d_k).float())
    grad2 = cal_grad(attn2)
    print( "after divide by sqrt(d_k), count of gradients that smaller than threshod({}) is {}, {:.4f}% \n".format(
        threshold, grad2.le(threshold).sum(), grad2.le(threshold).sum().item()/grad2.view(-1).size(0)*100 ) )