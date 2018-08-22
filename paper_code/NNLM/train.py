# Load packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from nnlm_data_loader import DataSet
from model import NNLM

def perplexity(x):
    return -torch.log(x).sum() / x.size(0)

def validation(model, loader):
    model.eval()
    pp = 0
    acc = 0
    for batch in loader:
        inputs, targets = batch[0], batch[1]
        preds = model.predict(inputs)
        probs, idxes = preds.max(1)
        acc += torch.eq(idxes, targets).sum().item()
        pp += perplexity(probs).item()
        
    return acc, pp/len(loader)

def train(config):
    USE_CUDA = torch.cuda.is_available()
    DEVICE = "cuda" if USE_CUDA else None
    # Create Dataset & Loader
    dataset_creator = DataSet(base_path=config.BASE_PATH, train=config.TRAIN_FILE, valid=config.VALID_FILE, test=config.TEST_FILE, n_gram=config.N_GRAM, tokenizer=str.split, save_tokens=False, direct_load=True, remove_short=True, device=DEVICE)
    train_data, valid_data, test_data = dataset_creator.splits()
    train_loader, valid_loader, test_loader= dataset_creator.create_loader(train=train_data, valid=valid_data, test=test_data, batch_size=config.BATCH)
    
    # Create Model
    V = len(train_data.vocab)    
    nnlm = NNLM(embed_size=config.EMBED, hidden_size=config.HIDDEN, vocab_size=V, num_prev_tokens=(config.N_GRAM-1))
    if USE_CUDA:
        nnlm = nnlm.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nnlm.parameters(), lr=config.LR, weight_decay=config.WD)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[int(config.STEP/2)], optimizer=optimizer)
    
    # Train
    start_time = time.time()
    for step in range(config.STEP):
        nnlm.train()
        scheduler.step()
        losses=[]
        for i, batch in enumerate(train_loader):
            inputs, targets = batch[0], batch[1]

            nnlm.zero_grad()

            outputs = nnlm(inputs)

            loss = loss_function(outputs, targets.view(-1))
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(nnlm.parameters(), 50.0)  # gradient clipping
            optimizer.step()

            if i % config.EVAL_EVERY == 0:
                msg = '[{}/{}][{}/{}] train_loss: {:.4f}'.format(step+1, config.STEP, i, len(train_loader), np.mean(losses))
                print(msg)
        # Validation
        acc_valid, pp_valid = validation(model=nnlm, loader=valid_loader)
        print('='*30)
        msg = '[{}/{}]\n valid_perplextiy: {:.4f} \n valid_accuracy: {:.4f}'.format(step+1, config.STEP, pp_valid, acc_valid/len(valid_data))
        print(msg)
        print('='*30)
    
    # Time 
    end_time = time.time()
    minute = int((end_time-start_time) // 60)
    print('Training Excution time with validation: {:d} m {:.4f} s'.format(minute, (end_time-start_time)-minute*60))
    
    # Test
    acc, pp = validation(model=nnlm, loader=test_loader)
    msg = 'test_perplextiy: {:.4f}, test_accuracy: {:.4f}'.format(pp, acc/len(test_data))
    print('='*30)
    print(msg)
    
    # Save Model
    torch.save(nnlm.state_dict(), config.SAVE_PATH+'({:.4f})'.format(pp))