import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """Label Smoothing"""
    def __init__(self, trg_vocab_size, pad_idx, eps=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        self.pad_idx = pad_idx
        self.eps = eps
        self.trg_vocab_size = trg_vocab_size
        self.true_dist = None
        
    def forward(self, x, target):
        """
        Inputs:
        x: (B*T_q, V_target), scores
        t: (B, T_q)
        ref: https://arxiv.org/pdf/1512.00567.pdf  
        #7
        q'(k | x) = (1 - eps) q(k) + eps * u(k)
        where: q(k) = 1 if k == 'target class' else 0
        """
        assert x.size(1) == self.trg_vocab_size, \
            'vocab size is not equal x: {}, vocab: {}'.format(x.size(1), self.trg_vocab_size)
        assert target.dim() == 2, 't must be size of (B, T_q)'
        
        if self.eps == 0.0:
            return self.criterion(x, target.view(-1))
        
        true_dist = torch.zeros_like(x)
        true_dist.scatter(1, target.view(-1, 1), 1.0)
        true_dist = (1 - self.eps) * true_dist + self.eps / (self.trg_vocab_size -1)
        log_prob = F.log_softmax(x, dim=1)
        loss = -(true_dist * log_prob).sum(1)
        non_pad_mask = t.view(-1).ne(self.pad_idx)
        
        return loss.masked_select(non_pad_mask).sum()
        
        
        # true_dist = x.clone()
        # u(k) = 1 / K, exclude token <s>, <pad>
        # true_dist.fill_(self.eps / (self.trg_vocab_size - 2))  
        # at target index, value is (1-eps) * 1
        # true_dist.scatter_(1, target.view(-1, 1), self.confidence)
        # true_dist[:, self.pad_idx] = 0  # exclude token <s>
        # mask = torch.nonzero(target.view(-1) == self.pad_idx)
        # if mask.dim() > 0:
        #    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # self.true_dist = true_dist
        # return self.criterion(x.log_softmax(1), true_dist.detach())    
    
#     def forward(self, x, target):
#         """
#         Inputs:
#         x: (B*T_q, V_target), scores
#         t: (B, T_q)
#         ref: https://arxiv.org/pdf/1512.00567.pdf  
#         #7
#         q'(k | x) = (1 - eps) q(k) + eps * u(k)
#         where: q(k) = 1 if k == 'target class' else 0
#         """
#         assert x.size(1) == self.trg_vocab_size, \
#             'vocab size is not equal x: {}, vocab: {}'.format(x.size(1), self.trg_vocab_size)
#         assert target.dim() == 2, 't must be size of (B, T_q)'
        
#         if self.eps == 0.0:
#             return self.cross_entropy(x, target.view(-1))
#         true_dist = x.clone()
#         # u(k) = 1 / K, exclude token <s>, <pad>
#         true_dist.fill_(self.eps / (self.trg_vocab_size - 2))  
#         # at target index, value is (1-eps) * 1
#         true_dist.scatter_(1, target.view(-1, 1), self.confidence)
#         true_dist[:, self.pad_idx] = 0  # exclude token <s>
#         mask = torch.nonzero(target.view(-1) == self.pad_idx)
#         if mask.dim() > 0:
#             true_dist.index_fill_(0, mask.squeeze(), 0.0)
#         self.true_dist = true_dist
#         return self.criterion(x.log_softmax(1), true_dist.detach())