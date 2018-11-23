import torch
import torch.nn.functional as F
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


def most_similar(word, word_vectors, vocab, top_k=10):
    sims = []
    wv1 = word_vectors[vocab.stoi[word]] 
    for i in range(len(vocab)):
        if vocab.itos[i] == word:
            continue
        wv2 = word_vectors[i]
        sim = F.cosine_similarity(torch.FloatTensor(wv1), torch.FloatTensor(wv2), dim=0)
        sims.append((vocab.itos[i], sim.item()))
    return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
