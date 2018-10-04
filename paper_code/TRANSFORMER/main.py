import argparse
import torch
from train import build_dataloader, build_model_optimizer, train_model

def main():
    parser = argparse.ArgumentParser(description='TRANSFORMER argument parser')
    # data path, basic settings and cuda
    parser.add_argument('-iwslt', '--IWSLT', help='Use iwslt dataset', action='store_true')
    parser.add_argument('-maxlen', '--MAX_LEN', help='Max length of sentence', type=int, default=100)
    parser.add_argument('-minfreq', '--MIN_FREQ', help='Minmum frequence of vocab', type=int, default=2)
    # custom dataset
    parser.add_argument('-pth_tr', '--TRAIN_PATH', help='location of path where your train data is located, must be tsv seperated in translation task')
    parser.add_argument('-pth_va', '--VALID_PATH', help='location of path where your valid data is located, must be tsv seperated in translation task')
    parser.add_argument('-exts', '--EXTS', help='"Source to Target" or "Target to source"', type=str, default='src-trg')
    parser.add_argument('-sos', '--SOS', help='Start of sentence token', default=None)
    parser.add_argument('-eos', '--EOS', help='End of sentence token', default=None)
    parser.add_argument('-stp', '--STEP', help='Trainging Steps', type=int, default=10)
    parser.add_argument('-bs', '--BATCH', help='Batch Size', type=int, default=64)
    parser.add_argument('-cuda', '--USE_CUDA', help='Use cuda if exists', action='store_true')
    parser.add_argument('-emptymem', '--EMPTY_CUDA_MEMORY', help='Use cuda empty cashce', action='store_true')
    parser.add_argument('-evalee', '--EVAL_EVERY', help='Eval Every', type=int, default=10)
    # model
    parser.add_argument('-nl', '--N_LAYER', help='Number of layers in Transformer Model', type=int, default=6)
    parser.add_argument('-nh', '--N_HEAD', help='Number of layers in Multihead Attention', type=int, default=8)
    parser.add_argument('-dk', '--D_K', help='Demension of keys', type=int, default=64)
    parser.add_argument('-dv', '--D_V', help='Demension of values', type=int, default=64)
    parser.add_argument('-dm', '--D_MODEL', help='Demension of model', type=int, default=512)
    parser.add_argument('-df', '--D_F', help='Demension of last linear layer', type=int, default=512)
    parser.add_argument('-drop', '--DROP_RATE', help='Dropout rate',type=float, default=0.1)
    parser.add_argument('-conv', '--USE_CONV', help='Use Convolution operation in PositionWise FeedForward Network', action='store_true')
    parser.add_argument('-attn', '--RETURN_ATTN', help='Return attentions', action='store_true')
    parser.add_argument('-lws', '--LINEAR_WS', help='Linear weight share with decoder embedding', action='store_true')
    parser.add_argument('-ews', '--EMBED_WS', help='Embed weight share between encoder and decoder', action='store_true')
    
    # eval & save model
    parser.add_argument('-thres', '--THRES', help='Threshold for earlystopping',type=float, default=0.9999)
    parser.add_argument('-eps', '--EPS', help='Label Smoothing',type=float, default=0.1)
    parser.add_argument('-warm', '--WARMUP', help='Warmup size in optimizer', type=int, default=4000)
    parser.add_argument('-save', '--SAVE_MODEL', help='Save model', action='store_true')
    parser.add_argument('-svp', '--SAVE_PATH', help='Model path', type=str, default='./model/')
    parser.add_argument('-savebest', '--SAVE_BEST', help='Save model', action='store_true')
    
    config = parser.parse_args()
    print(config)
    if config.USE_CUDA:
        assert config.USE_CUDA == torch.cuda.is_available(), 'cuda is not avaliable.'
    DEVICE = 'cuda' if config.USE_CUDA else None
    if config.IWSLT:
        SRC, TRG, train_loader, valid_loader = build_dataloader(config)
        model, optimizer, loss_function = build_model_optimizer(config, (SRC, TRG), device=DEVICE)
        train_model(train_loader, valid_loader, model, optimizer, loss_function, config, device=DEVICE)
    else:
        train, valid, train_loader, valid_loader = build_dataloader(config)
        model, optimizer, loss_function = build_model_optimizer(config, train, device=DEVICE)
        train_model(train_loader, valid_loader, model, optimizer, loss_function, config, device=DEVICE)
    
if __name__ == '__main__':
    main()