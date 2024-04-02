import pandas as pd
import torchtext

from dataLoader import clean_data, tokenization


if __name__ == "__main__":
    # df = pd.read_csv("singapore_airlines_reviews.csv")

    # Clean Data and save as data_clean.csv
    # clean_data(df)

    df = pd.read_csv("data_clean.csv")
    paragraphs = df['text']
    
    # Convert words into numerical representation using GloVe
    GloVe = torchtext.vocab.GloVe(name="840B", dim=300)

    tokenized_data = tokenization(paragraphs, glove=GloVe, lowercase=True, stop_words=True)
    print(tokenized_data)