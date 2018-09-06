import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.embedding_w = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)

        init = (2.0 / (vocab_size + embed_size))**0.5  # Xavier init
        nn.init.uniform_(self.embedding_w.weight, -init, init)
        nn.init.uniform_(self.embedding_u.weight, -0.0, 0.0)

    def forward(self, inputs, targets, neg_samples):
        embed = self.embedding_w(inputs)  # B, 1, embed_size
        context = self.embedding_u(targets)  # B, 1, embed_size
        negs = -self.embedding_u(neg_samples)  # B, k, embed_size

        pos = context.bmm(embed.transpose(1, 2)).squeeze(2)  # B, 1
        neg = negs.bmm(embed.transpose(1, 2)).sum(1)  # B, k, 1  > B, 1
        nll = F.logsigmoid(pos) + F.logsigmoid(neg)
        return -torch.mean(nll)

    def cosine_similarity(self, idx1, idx2):
        wv1 = self.embedding_w.weight[idx1]
        wv2 = self.embedding_w.weight[idx2]
        return F.cosine_similarity(wv1, wv2, dim=0)
