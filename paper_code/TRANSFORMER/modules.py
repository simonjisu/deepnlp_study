import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = torch.tensor(d_k).float()
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, mask=None):
        """
        Inputs:
        * q: (B, T_q, d_q), d_q = d_k
        * k: (B, T_k, d_k)
        * v: (B, T_v, d_v), T_k = T_v
        -------------------------------
        Outputs:
        * output: (B, T_q, d_v)
        * probs: (B, T_q, T_k)
        """
        assert q.size(2) == k.size(2), "d_q = d_k"
        assert k.size(1) == v.size(1), "T_k = T_v"
        attn = torch.bmm(q, k.transpose(1, 2))  # (B, T_q, d_k) * (B, T_k, d_k) -> (B, T_q, T_k)
        attn = attn / torch.sqrt(self.d_k)
        # why doing this? 
        # for the large values of d_k, the dot products grow large in magnitude, 
        # pushing the softmax function into regions where it has extremely small gradients
        # to counteract this effect, scaled the dot products by 1/sqrt(d_k)
        # to illustrate why the dot products get large,
        # check the function 'check_dotproduct_dist'
        if mask is not None:
            attn = attn.masked_fill_(mask, -np.inf)
        
        attn = self.softmax(attn)  # (B, T_q, T_k) --> (B, T_q, T_k)
        output = torch.bmm(attn, v)  # (B, T_q, T_k) * (B, T_v, d_v) --> (B, T_q, d_v), make sure that T_k == T_v
        return output, attn
    
class XavierLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)