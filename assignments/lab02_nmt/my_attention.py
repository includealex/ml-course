import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time

from my_network import Encoder
from my_network import Seq2Seq

class AttentionEncoder(Encoder):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__(input_dim, emb_dim, hid_dim, n_layers, dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        
        return output, hidden, cell

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        self.softmax = nn.Softmax(dim=2)
        
        self.rnn = nn.LSTM(
            input_size=emb_dim+n_layers*hid_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.out = nn.Linear(hid_dim, output_dim)
                
    def forward(self, input, enc_hidden, dec_hidden, dec_cell):
 
        batch_size = dec_hidden.shape[1]
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        hidd_dot_prod = torch.einsum("sbh, nbh -> bns", enc_hidden, dec_hidden)
        hidd_coeff = self.softmax(hidd_dot_prod)
        attention_hidd = torch.einsum("sbh, bns -> bnh", enc_hidden, hidd_coeff)
        attention_hidd = attention_hidd.reshape(1, batch_size, -1)

        modified_input = torch.cat([attention_hidd, embedded], dim=-1)

        output, (hidden, cell) = self.rnn(modified_input, (dec_hidden, dec_cell))
        prediction = self.out(output.squeeze(0))        
        return prediction, hidden, cell
    
class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, encoder, decoder, device):
        super().__init__(encoder, decoder, device)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)       
        enc_output, dec_hidden, dec_cell = self.encoder(src)       
        input = trg[0,:]
        for t in range(1, max_len):
            output, dec_hidden, dec_cell = self.decoder(input, 
                                                        enc_output, 
                                                        dec_hidden, 
                                                        dec_cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(dim=1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs