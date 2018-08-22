import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NNLM argument parser\n download the data from "https://github.com/e9t/nsmc"')
    parser.add_argument('-pth', '--BASE_PATH', help='location of base_path where your data is located, required', type=str, required=True)
    parser.add_argument('-trp', '--TRAIN_FILE', help='location of training path, default is "train_tokens"', type=str, default='train_tokens')
    parser.add_argument('-vap', '--VALID_FILE', help='location of valid path, default is "valid_tokens"', type=str, default='valid_tokens')
    parser.add_argument('-tep', '--TEST_FILE', help='location of test path, default is "test_tokens"', type=str, default='test_tokens')
    parser.add_argument('-svp', '--SAVE_PATH', help='saving model path, default is "./model/nnlm.model"', type=str, default='./model/nnlm.model')
    parser.add_argument('-bat', '--BATCH', help='batch size, default is 1024', type=int, default=1024)
    parser.add_argument('-ngram', '--N_GRAM', help='n-gram size, default is 5', type=int, default=5)
    parser.add_argument('-hid', '--HIDDEN', help='hidden size, default is 500', type=int, default=500)
    parser.add_argument('-emd', '--EMBED', help='embed size, default is 100', type=int, default=100)
    parser.add_argument('-stp', '--STEP', help='number of iteration, default is 5', type=int, default=5)
    parser.add_argument('-lr', '--LR', help='learning rate, default is 0.001', type=float, default=0.001)
    parser.add_argument('-wdk', '--WD', help='L2 regularization, weight_decay in optimizer, default is 10e-5', type=float, default=0.00001)
    parser.add_argument('-ee', '--EVAL_EVERY', help='eval every batch size, default is 1000', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    train(config)