# coding: utf-8
# author: simonjisu
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata

from models import Transformer
from trans_dataloader import TranslateDataset
from schoptim import ScheduledOptim
            
def cal_loss(pred, target, smoothing, pad_idx=1):
    """
    Calculate cross entropy loss, apply label smoothing if needed. 
    borrowed from: 
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
    """
    target = target.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, target, ignore_index=pad_idx, reduction='sum')
    return loss


def run_step(loader, model, optimizer, smoothing, device=None):
    model = model.to(device)
    model.train()
    loss_per_step = 0
    eval_every = len(loader) // 5
    for i, batch in enumerate(loader):
        src, src_pos, trg, trg_pos = map(lambda x: x.to(device), batch)
        model.zero_grad()
        # forward
        output = model(enc=src, enc_pos=src_pos, dec=trg, dec_pos=trg_pos)
        # eval
        pred = output.cpu()
        target = trg.cpu().view(-1)
        loss = cal_loss(pred, target, smoothing, pad_idx=model.pad_idx)
        loss.backward()        
        # update parameters
        optimizer.step_and_update_lr()
        total_words = target.ne(model.pad_idx).sum().item()
        loss_per_step += loss.item() / total_words
        if i % eval_every == 0:
            print(' > [{}/{}] loss_per_batch: {:.4f}'.format(i, len(loader), loss.item()))
    return loss_per_step


def validation(loader, model, smoothing=False, device=None):
    model.eval()
    loss_per_step = 0
    
    for i, batch in enumerate(loader):
        src, src_pos, trg, trg_pos = map(lambda x: x.to(device), batch)
        model.zero_grad()
        # forward
        output = model(enc=src, enc_pos=src_pos, dec=trg, dec_pos=trg_pos)
        # eval
        pred = output.cpu()
        target = trg.cpu().view(-1)
        loss = cal_loss(pred, target, smoothing, pad_idx=model.pad_idx)

        total_words = target.ne(model.pad_idx).sum().item()
        loss_per_step += loss.item() / total_words
    return loss_per_step


def build_model_optimizer(config, train, device=None):
    model = Transformer(enc_vocab_len=len(train.src_vocab.stoi),
                    enc_max_seq_len=train.src_maxlen, 
                    dec_vocab_len=len(train.trg_vocab.stoi), 
                    dec_max_seq_len=train.trg_maxlen, 
                    n_layer=config.N_LAYER, 
                    n_head=config.N_HEAD, 
                    d_model=config.D_MODEL, 
                    d_k=config.D_K,
                    d_v=config.D_V,
                    d_f=config.D_F, 
                    pad_idx=train.src_vocab.stoi['<pad>'],
                    drop_rate=config.DROP_RATE, 
                    use_conv=config.USE_CONV, 
                    return_attn=config.RETURN_ATTN,
                    linear_weight_share=config.LINEAR_WS, 
                    embed_weight_share=config.EMBED_WS).to(device)
    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           betas=(0.9, 0.98), eps=1e-09), 
                           config.D_MODEL, 
                           config.WARMUP)
    return model, optimizer

def build_dataloader(config):
    train = TranslateDataset(path=config.TRAIN_PATH, exts=config.EXTS)
    valid = TranslateDataset(path=config.VALID_PATH, 
                             vocab=[('src', train.src_vocab), ('trg', train.trg_vocab)])
    train_loader = torchdata.DataLoader(dataset=train,
                                    collate_fn=train.collate_fn,
                                    batch_size=config.BATCH, 
                                    shuffle=True, 
                                    drop_last=False)
    valid_loader = torchdata.DataLoader(dataset=valid,
                                        collate_fn=valid.collate_fn,
                                        batch_size=config.BATCH, 
                                        shuffle=True, 
                                        drop_last=False)
    return train, valid, train_loader, valid_loader


def train_model(train_loader, valid_loader, model, optimizer, config, device=None):
    start_time = time.time()
    valid_losses = []
    for step in range(config.STEP):
        print("--"*20)
        train_loss = run_step(train_loader, model, optimizer, 
                              smoothing=config.SMOOTHING, device=device)
        torch.cuda.empty_cache()
        valid_loss = validation(valid_loader, model, smoothing=False, device=device)
        torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        print('[{}/{}] train: {:.4f}, valid: {:.4f} \n'.format(
            step+1, config.STEP, train_loss, valid_loss))
        
        if config.SAVE_MODEL:
            model_path = config.SAVE_PATH + '{}_{:.4f}'.format(step, valid_loss)
            torch.save(model.cpu().state_dict(), model_path)
            print('****** model saved updated! ******')
        
            if valid_loss < 0.00000001:
                print('****** early break!! ******')
                breaks

    end_time = time.time()
    total_time = end_time-start_time
    hour = int(total_time // (60*60))
    minute = int((total_time - hour*60*60) // 60)
    second = total_time - hour*60*60 - minute*60
    print('\nTraining Excution time with validation: {:d} h {:d} m {:.4f} s'.format(hour, minute, second))
    