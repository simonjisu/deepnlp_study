import torch
import torch.optim as optim
import torch.utils.data as torchdata
from collections import Counter
import numpy as np
import time
import sys

from model import Word2Vec
from word2vec_utils import negative_sampling
from word2vec_dataloader import CustomDataset
from preprocessing import preprocess


def train(config):
    datas = preprocess(config.PATH)
    if config.N_WORDS:
        datas = [datas[0][:config.N_WORDS]]
    USE_CUDA = torch.cuda.is_available()
    DEVICE = 'cuda' if USE_CUDA else None

    train_data = CustomDataset(datas, window=config.WINDOW_SIZE, device=DEVICE)
    train_loader = torchdata.DataLoader(train_data, batch_size=config.BATCH, shuffle=True, collate_fn=train_data.collate_fn)
    print('number of total pair datasets {}'.format(len(train_data)))
    # build unigram_distribution for negative sampling
    vocab_counter = Counter(train_data.flatten(datas))
    total_words = len(vocab_counter)
    unigram_distribution = []
    for w, c in vocab_counter.items():
        unigram_distribution.extend([w] * int(((c / total_words)**(3 / 4)) / config.Z))
    if int(((1 / total_words)**(3 / 4)) / config.Z) == 0:
        print('WARNING: building unigram-distribution for a single word is ', int(((1 / total_words)**(3 / 4)) / config.Z))
        print('please reset or control Z to a smaller number when your vocab is too large')
    print('total unique words: {}, number of words in unigram-distribution {}'.format(total_words, len(unigram_distribution)))
    unigram_distribution = train_data._numerical(unigram_distribution)

    # build model
    V = len(train_data.vocab)
    model = Word2Vec(V, config.EMBED)
    if USE_CUDA:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[20, 40], optimizer=optimizer)

    # train
    losses = []
    start_time = time.time()
    for step in range(config.STEP):
        scheduler.step()
        for i, batch in enumerate(train_loader):
            inputs, targets = batch
            neg_samples = negative_sampling(targets, unigram_distribution, config.K, use_cuda=USE_CUDA)

            model.zero_grad()
            loss = model(inputs, targets, neg_samples)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        if step % config.EVAL_EVERY == 0:
            msg = "[{}/{}] loss {:.4f}".format(step + 1, config.STEP, np.mean(losses))
            print(msg)
            losses = []

    end_time = time.time()
    minute = int((end_time - start_time) // 60)
    print('Training Excution time with validation: {:d} m {:.4f} s'.format(minute, (end_time - start_time) - minute * 60))
    # save
    torch.save(model.state_dict(), config.SAVE_PATH)
