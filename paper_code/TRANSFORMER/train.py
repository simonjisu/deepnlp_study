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
from lbs import LabelSmoothing
#
from torchtext.data import Field, Iterator
from torchtext import datasets
import spacy


def get_pos(x, pad_idx=1, pospad_idx=0):
    B, T = x.size()
    pos = torch.arange(1, T+1).expand(B, -1)
    padding_mask = (x == pad_idx)
    pos = pos.masked_fill(padding_mask, pospad_idx)
    return pos


def cal_performance(pred, target, loss_function, pad_idx=0):
    loss = loss_function(pred, target)

    pred = pred.max(1)[1]
    non_pad_mask = target.view(-1).ne(pad_idx)
    n_correct = pred.eq(target.view(-1))
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def run_step(loader, model, optimizer, loss_function, config, device=None):
    model = model.to(device)
    model.train()
    loss_per_step = 0
    total_words = 0
    correct_words = 0
    eval_every = len(loader) // config.EVAL_EVERY
    
    start_time = time.time()
    for i, batch in enumerate(loader):
        if config.IWSLT:
            src_pos, trg_pos = map(get_pos, [batch.src, batch.trg])
            batch = [batch.src, src_pos, batch.trg, trg_pos]
        src, src_pos, trg, trg_pos = map(lambda x: x.to(device), batch)
        model.zero_grad()
        # forward
        output = model(enc=src, enc_pos=src_pos, dec=trg, dec_pos=trg_pos)
        # loss and backward
        pred = output.cpu()
        target = trg.cpu()
        loss, n_correct = cal_performance(pred, target, loss_function, pad_idx=model.pad_idx)
        loss.backward()        
        # update parameters
        optimizer.step_and_update_lr()
        
        # eval
        n_words = target.view(-1).ne(model.pad_idx).sum().item()
        total_words += n_words
        correct_words += n_correct
        loss_per_step += loss.item()

        if i % eval_every == 0:
            end_time = time.time()
            total_time = end_time-start_time
            print(' > [{}/{:.2f}] loss_per_batch: {:.4f} time: {:.1f} s'.format(i, i/len(loader)*100, loss.item()/n_words, total_time))
            start_time = time.time()
            
    accuracy = correct_words / total_words
    loss_per_step = loss_per_step / total_words
    return loss_per_step, accuracy


def validation(loader, model, loss_function, config, device=None):
    model.eval()
    loss_per_step = 0
    total_words = 0
    correct_words = 0
    for i, batch in enumerate(loader):
        if config.IWSLT:
            src_pos, trg_pos = map(get_pos, [batch.src, batch.trg])
            batch = [batch.src, src_pos, batch.trg, trg_pos]
        src, src_pos, trg, trg_pos = map(lambda x: x.to(device), batch)
        model.zero_grad()
        # forward
        output = model(enc=src, enc_pos=src_pos, dec=trg, dec_pos=trg_pos)
        # loss and backward
        pred = output.cpu()
        target = trg.cpu()
        loss, n_correct = cal_performance(pred, target, loss_function, pad_idx=model.pad_idx)
        # eval
        n_words = target.view(-1).ne(model.pad_idx).sum().item()
        total_words += n_words
        correct_words += n_correct
        loss_per_step += loss.item()
    
    
    accuracy = correct_words / total_words
    loss_per_step = loss_per_step / total_words
    
    
    return loss_per_step, accuracy


def build_model_optimizer(config, train, device=None):
    if config.IWSLT:
        src_field, trg_field = train
        model = Transformer(enc_vocab_len=len(src_field.vocab.stoi),
                        enc_max_seq_len=config.MAX_LEN, 
                        dec_vocab_len=len(trg_field.vocab.stoi), 
                        dec_max_seq_len=config.MAX_LEN, 
                        n_layer=config.N_LAYER, 
                        n_head=config.N_HEAD, 
                        d_model=config.D_MODEL, 
                        d_k=config.D_K,
                        d_v=config.D_V,
                        d_f=config.D_F, 
                        pad_idx=src_field.vocab.stoi['<pad>'],
                        drop_rate=config.DROP_RATE, 
                        use_conv=config.USE_CONV, 
                        return_attn=config.RETURN_ATTN,
                        linear_weight_share=config.LINEAR_WS, 
                        embed_weight_share=config.EMBED_WS).to(device)
        optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           betas=(0.9, 0.98), eps=1e-09), 
                           config.D_MODEL, 
                           config.WARMUP)
        loss_function = LabelSmoothing(trg_vocab_size=len(trg_field.vocab.stoi), 
                                       pad_idx=trg_field.vocab.stoi['<pad>'], 
                                       eps=config.EPS)
        return model, optimizer, loss_function
    else:
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
    loss_function = LabelSmoothing(trg_vocab_size=len(train.trg_vocab.stoi), 
                                   pad_idx=train.src_vocab.stoi['<pad>'], 
                                   eps=config.EPS)
    return model, optimizer, loss_function


def build_dataloader(config):
    if config.IWSLT:
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(tokenize=tokenize_de, lower=True, batch_first=True)
        TRG = Field(tokenize=tokenize_en, init_token=config.SOS, eos_token=config.EOS, lower=True, batch_first=True)

        train, valid, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TRG), 
           root=config.TRAIN_PATH, filter_pred=lambda x: len(vars(x)['src']) <= config.MAX_LEN and len(vars(x)['trg']) <= config.MAX_LEN)
        SRC.build_vocab(train.src, min_freq=config.MIN_FREQ)
        TRG.build_vocab(train.trg, min_freq=config.MIN_FREQ)
        train_loader, valid_loader = Iterator.splits(datasets=(train, valid),
                                                     batch_sizes=(config.BATCH, config.BATCH), 
                                                     shuffle=True, repeat=False)
        return SRC, TRG, train_loader, valid_loader
    
    else:    
        train = TranslateDataset(path=config.TRAIN_PATH, exts=config.EXTS, sos=config.SOS, eos=config.EOS)
        valid = TranslateDataset(path=config.VALID_PATH, sos=config.SOS, eos=config.EOS,
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


def train_model(train_loader, valid_loader, model, optimizer, loss_function, config, device=None):
    start_time = time.time()
    valid_losses = [9999]
    for step in range(config.STEP):
        print("--"*20)
        train_loss, train_acc = run_step(train_loader, model, optimizer, loss_function, config, device=device)
        if config.EMPTY_CUDA_MEMORY:
            torch.cuda.empty_cache()
        valid_loss, valid_acc = validation(valid_loader, model, loss_function, config, device=device)
        if config.EMPTY_CUDA_MEMORY:
            torch.cuda.empty_cache()
        valid_losses.append(valid_loss)

        print('[{}/{}] (train) loss {:.4f}, acc {:.4f} | (valid) loss {:.4f}, acc {:.4f} \n'.format(
                step+1, config.STEP, train_loss, train_acc, valid_loss, valid_acc))

        # Save model
        if config.SAVE_MODEL:
            if config.SAVE_BEST:
                if valid_loss <= min(valid_losses):
                    torch.save(model.state_dict(), config.SAVE_PATH)
                    print('****** model saved updated! ******')

                if valid_acc > config.THRES:
                    print('****** early break!! ******')
                    break
            else:
                model_path = config.SAVE_PATH + '{}_{:.4f}_{:.4f}'.format(step, train_acc, valid_acc)
                torch.save(model.state_dict(), model_path)
                print('****** model saved updated! ******')

                if valid_acc > config.THRES:
                    print('****** early break!! ******')
                    break

    end_time = time.time()
    total_time = end_time-start_time
    hour = int(total_time // (60*60))
    minute = int((total_time - hour*60*60) // 60)
    second = total_time - hour*60*60 - minute*60
    print('\nTraining Excution time with validation: {:d} h {:d} m {:.4f} s'.format(hour, minute, second))
    