import torch
import torch.nn as nn

class my_model(nn.Module):
    
    def __init__(self, vocab_size, emb_size, hidden_size, padding_idx):
        super(my_model, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)#设置词嵌入，并设置padding为0
        
        
    