# package load
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess
from word2vec_dataloader import CustomDataset
from word2vec_utils import most_similar
from model import Word2Vec

def main():
    USE_CUDA = False
    DEVICE = 'cuda' if USE_CUDA else None
    WINDOW_SIZE = 2
    N_WORDS = 10000000 # only trained first 10,000,000 words

    datas = [preprocess('../../data/wiki/enwik9_text')[0][:N_WORDS]]
    train_data = CustomDataset(datas, window=WINDOW_SIZE, device=DEVICE)
    V = len(train_data.vocab)
    EMBED = 300
    model = Word2Vec(V, EMBED)
    if USE_CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load('./model/word2vec_wiki.model'))

    features = model.embedding_w.weight.detach().numpy()
    tsne = TSNE(learning_rate=200)
    transformed = tsne.fit_transform(features)

    np.save('./model/t_sne.npy', transformed)
    print('done!')

if __name__ == "__main__":
    main()