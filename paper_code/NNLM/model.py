import torch
import torch.nn as nn
import torch.nn.functional as F

class NNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_prev_tokens):
        super(NNLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_prev_tokens = num_prev_tokens
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
            nn.Linear(num_prev_tokens*embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, inputs):
        embeded = self.embed(inputs)
        concated = torch.cat([x for x in embeded.transpose(0, 1)], dim=1)
        output = self.fc(concated)
        return output
    
    def predict(self, inputs):
        output = self.forward(inputs)
        return F.softmax(output, 1)