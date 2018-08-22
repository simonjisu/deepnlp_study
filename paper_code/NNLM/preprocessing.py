import argparse
from konlpy.tag import Twitter
from nnlm_data_loader import DataSet

def preprocess(config):
    TAGGER = lambda x: ['/'.join(y) for y in Twitter().pos(x, norm=True)]
    train, valid, test = DataSet(base_path=config.BASE_PATH, train=config.TRAIN_FILE, valid=config.VALID_FILE, test=config.TEST_FILE, n_gram=config.N_GRAM, tokenizer=TAGGER, save_tokens=True, direct_load=False).splits()
    
    print('Done!')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocessing argparser')
    parser.add_argument('-pth', '--BASE_PATH', help='location of base_path where your data is located, required', type=str, required=True)
    parser.add_argument('-trp', '--TRAIN_FILE', help='location of training path, default is "train_tokens"', type=str, default='train_tokens')
    parser.add_argument('-vap', '--VALID_FILE', help='location of valid path, default is "valid_tokens"', type=str, default='valid_tokens')
    parser.add_argument('-tep', '--TEST_FILE', help='location of test path, default is "test_tokens"', type=str, default='test_tokens')
    parser.add_argument('-ngram', '--N_GRAM', help='n-gram size, default is 5', type=int, default=5)
    config = parser.parse_args()
    preprocess(config)
