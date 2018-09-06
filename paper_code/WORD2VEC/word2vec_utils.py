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


def most_similar(word, model, vocab, top_k=10):
    sims = []
    for i in range(len(vocab)):
        if vocab.itos[i] == word:
            continue
        sim = model.cosine_similarity(vocab.stoi[word], i)
        sims.append((vocab.itos[i], sim.item()))
    return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
