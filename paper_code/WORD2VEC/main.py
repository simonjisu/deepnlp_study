import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WORD2VEC argument parser')
    parser.add_argument('-pth', '--PATH', help='location of path where your data is located, default is "brown", it will use nltk, brown corpus', type=str, default='brown')
    parser.add_argument('-svp', '--SAVE_PATH', help='saving model path, default is "./model/word2vec.model"', type=str, default='./model/word2vec.model')
    parser.add_argument('-bat', '--BATCH', help='batch size, default is 1024', type=int, default=1024)
    parser.add_argument('-win', '--WINDOW_SIZE', help='window size, default is 2', type=int, default=2)
    parser.add_argument('-z', '--Z', help='normalization constant for negative sampling, default is 1e-4', type=float, default=1e-4)
    parser.add_argument('-k', '--K', help='sampling number, default is 10', type=int, default=10)
    parser.add_argument('-emd', '--EMBED', help='embed size, default is 300', type=int, default=300)
    parser.add_argument('-stp', '--STEP', help='number of iteration, default is 60', type=int, default=60)
    parser.add_argument('-lr', '--LR', help='learning rate, default is 0.01', type=float, default=0.01)
    parser.add_argument('-ee', '--EVAL_EVERY', help='eval every batch size, default is 5', type=int, default=5)
    parser.add_argument('-n', '--N_WORDS', help='number words to use in wiki because memory cannot hold all tokens, default is 0', type=int, default=0)
    config = parser.parse_args()
    print(config)
    train(config)
