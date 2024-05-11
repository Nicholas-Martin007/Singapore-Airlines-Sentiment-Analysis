import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import time, sys, os

import torchtext


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        num_classes => outputs of the prediction which is ratings (1, 2, 3, 4, 5)?
        LSTM input size = 300 because of glove embedding dimension?

        """
        super(LSTMModel, self).__init__()

        # Embeddings
        glove = torchtext.vocab.GloVe(name="6B", dim=input_size)
        self.emb = nn.Embedding.from_pretrained(embeddings=glove.vectors, freeze=True) # https://discuss.huggingface.co/t/the-point-of-using-pretrained-model-if-i-dont-freeze-layers/40675
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.name = "LSTM_Model"

        # self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # * 2 based on how many layers
        # self.fc = nn.Linear(hidden_size * 2, 1024)
        # self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = self.emb(x)

        hidden_state = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        cell_state = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (hidden_state, cell_state))


        # Batch size, last timesteps, hidden_size
        out = out[:, -1, :]
        
        # out = F.relu(self.fc(out))
        # out = F.relu(self.fc1(out))
        # out = self.dropout(out)
        out = self.fc2(out)

        return out


