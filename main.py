import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchtext
import numpy as np

from dataLoader import clean_data, getDataLoader, retrieve_data
from model import LSTMModel
from test import test_sentiment_analysis
from train import train_sentiment_analysis

def train_pipeline(data, batch_size, glove, val_size, test_size):
    return

if __name__ == "__main__":

    # ============================

    # Reading + Cleaning Data (Using it for one-time use)
    # Using GloVe pre-trained word embeddings
    # Tokenizing Data
    # Data Loader (Using it depends on batch size, val_size, test_size)

    # ============================

    # df = pd.read_csv("singapore_airlines_reviews.csv")
    # clean_data(df)


    df = pd.read_csv("data_clean.csv")
    batch_size = 64
    val_size = 0.4
    test_size = 0.2
    GloVe = torchtext.vocab.GloVe(name="6B", dim=300)

    # train_loader, val_loader, test_loader = getDataLoader(data=df, batch_size=batch_size, glove=GloVe, val_size=val_size, test_size=test_size)

    train_loader = pickle.load(open('train_loader.pkl', 'rb'))
    val_loader = pickle.load(open('val_loader.pkl', 'rb'))
    test_loader = pickle.load(open('test_loader.pkl', 'rb'))

    model = LSTMModel(input_size=300, hidden_size=128, num_layers=2, num_classes=2)



    train_model = 0
    if train_model:
        train_sentiment_analysis(model, train_loader, val_loader, batch_size, lr=0.002, epochs=100)
    

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model_name = "model_LSTM_Model_bs_64_lr_0.002_epoch_12"
    model.load_state_dict(torch.load(model_name, map_location=device))
    test_sentiment_analysis(model, test_loader)
