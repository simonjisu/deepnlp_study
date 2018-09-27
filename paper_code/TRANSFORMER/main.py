import argparse
import torch
from train import build_dataloader, build_model_optimizer, train_model

def main():
    parser = argparse.ArgumentParser(description='TRANSFORMER argument parser')
    # data path, basic settings and cuda
    parser.add_argument('-pth_tr', '--TRAIN_PATH', help='location of path where your train data is located, must be tsv seperated in translation task', required=True)
    parser.add_argument('-pth_va', '--VALID_PATH', help='location of path where your valid data is located, must be tsv seperated in translation task', required=True)
    parser.add_argument('-exts', '--EXTS', help='"Source to Target" or "Target to source"', type=str, default='src-trg')
    parser.add_argument('-stp', '--STEP', help='Trainging Steps', type=int, default=10)
    parser.add_argument('-bs', '--BATCH', help='Batch Size', type=int, default=64)
    parser.add_argument('-cuda', '--USE_CUDA', help='Use cuda if exists', action='store_true')
    
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
    parser.add_argument('-smooth', '--SMOOTHING', help='Label smoothing', action='store_true')
    parser.add_argument('-warm', '--WARMUP', help='Warmup size in optimizer', type=int, default=4000)
    parser.add_argument('-save', '--SAVE_MODEL', help='Save model', action='store_true')
    parser.add_argument('-svp', '--SAVE_PATH', help='Model path', type=str, default='./model/')
    
    config = parser.parse_args()
    print(config)
    if config.USE_CUDA:
        assert config.USE_CUDA == torch.cuda.is_available(), 'cuda is not avaliable.'
    DEVICE = 'cuda' if config.USE_CUDA else None
    train, valid, train_loader, valid_loader = build_dataloader(config)
    model, optimizer = build_model_optimizer(config, train, device=DEVICE)
    train_model(train_loader, valid_loader, model, optimizer, config, device=DEVICE)
    
if __name__ == '__main__':
    main()