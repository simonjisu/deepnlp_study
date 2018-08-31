import torch
import random

def negative_sampling(targets, unigram_distribution, k, use_cuda=False):
    """
    unigram_distribution: have to be numericalize words
    """
    neg_samples = []
    batch_size = targets.size(0)
    for t in targets.view(-1).cpu().tolist():
        samples = []
        while len(samples) < k:
            word = random.choice(unigram_distribution)
            assert isinstance(word, int), "have to be numericalize words"
            if word == t:
                continue
            samples.append(word)
        neg_samples.append(samples)
    if use_cuda:
        return torch.LongTensor(neg_samples).cuda()
    return torch.LongTensor(neg_samples)