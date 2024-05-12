import pickle
import re
from nltk.corpus import wordnet
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torchtext
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


import torch

def clean_data(df):
    '''
    Removing unnecessary data columns for training, then saved the data
    '''

    drop_columns = ['published_date', 'published_platform', 'type', 'title', 'helpful_votes']

    df.drop(columns=drop_columns, inplace=True)
    
    # Title has 1 NaN value, need to fillna
    df = df.fillna("")

    df.to_csv('data_clean.csv', index=False)

def remove_emoji(text):
    '''
    Removing Emojis
    '''

    emojis = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002500-\U00002BEF" 
                    u"\U00002702-\U000027B0"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    u"\U0001f926-\U0001f937"
                    u"\U00010000-\U0010ffff"
                    u"\u2640-\u2642"
                    u"\u2600-\u2B55"
                    u"\u200d"
                    u"\u23cf"
                    u"\u23e9"
                    u"\u231a"
                    u"\ufe0f"
                    u"\u3030"
                    "]+", flags=re.UNICODE)
    return emojis.sub(r'', text)


def get_pos(tag):
    dictionary = {'J': wordnet.ADJ,
                  'V': wordnet.VERB,
                  'N': wordnet.NOUN,
                  'R': wordnet.ADV}
    return dictionary.get(tag[0], wordnet.NOUN)


def retrieve_data(data, glove, lowercase=True, stop_words=True):
    '''
    Tokenize for 1 Paragraph
    '''
    tokenized_data = []
    # labels = []
    
    def tokenization():
        for paragraph, rating in zip(data['text'], data['rating']):
            if lowercase:
                paragraph = paragraph.lower()

            # Remove words
            paragraph = remove_emoji(paragraph)

            # Split paragraph into bunch of words
            tokens = paragraph.split()

            # Removing punctuation
            paragraph = paragraph.translate(str.maketrans('', '', string.punctuation))

            # Removing stop words
            if stop_words:
                stops = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stops]
            
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word=token, pos=get_pos(tag)) for token, tag in pos_tag(tokens)]

            # Inserting into glove word embeddings
            tokenized_paragraph = []
            for token in tokens:
                if token and token in glove.stoi:
                    tokenized_paragraph.append(glove.stoi[token])
                else:
                    tokenized_paragraph.append(glove.stoi['unk'])

            # Truncate + Padding
            if len(tokenized_paragraph) < 256:
                tokenized_paragraph += [glove.stoi['pad']] * (256 - len(tokenized_paragraph))
            elif len(tokenized_paragraph) > 256:
                tokenized_paragraph = tokenized_paragraph[:256]

            if rating <= 3: rating = 0
            else: rating = 1

            tokenized_data.append((torch.tensor(tokenized_paragraph), rating))
            # labels.append(rating)

    tokenization()

    return tokenized_data
    # return list(zip(tokenized_data, labels))


def split_data(data, val_size, test_size):
    split_val = int(val_size * len(data))
    split_test = int(test_size * len(data))

    train_data, val_temp = data[split_val:], data[:split_val]
    val_data, test_data = val_temp[split_test:], val_temp[:split_test]
    

    return train_data, val_data, test_data



def getDataLoader(data, batch_size, glove, val_size, test_size):
    # tokenization
    tokenized_data = retrieve_data(data=data, glove=glove)
    
    # split data
    train_data, val_data, test_data = split_data(tokenized_data, val_size, test_size)


    train_sampler = SubsetRandomSampler(range(len(train_data)))
    val_sampler = SubsetRandomSampler(range(len(val_data)))
    test_sampler = SubsetRandomSampler(range(len(test_data)))


    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)    

    pickle.dump(train_loader, open('train_loader.pkl', 'wb'))
    pickle.dump(val_loader, open('val_loader.pkl', 'wb'))
    pickle.dump(test_loader, open('test_loader.pkl', 'wb'))

    return train_loader, val_loader, test_loader