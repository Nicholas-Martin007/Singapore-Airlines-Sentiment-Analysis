import pandas as pd
from sklearn.model_selection import train_test_split
import torchtext
import numpy as np

from dataLoader import clean_data, getDataLoader, retrieve_data

def train_pipeline(data, batch_size, glove, val_size, test_size):
    train_loader, val_loader, test_loader = getDataLoader(data, batch_size, glove, val_size, test_size)


if __name__ == "__main__":

    # Reading + Cleaning Data (Using it for one-time use)
    # Using GloVe pre-trained word embeddings
    # Tokenizing Data


    # df = pd.read_csv("singapore_airlines_reviews.csv")
    # clean_data(df)


    df = pd.read_csv("data_clean.csv")

    GloVe = torchtext.vocab.GloVe(name="6B", dim=300)

    train_pipeline(data=df, batch_size=256, glove=GloVe, val_size=0.35, test_size=0.1)
