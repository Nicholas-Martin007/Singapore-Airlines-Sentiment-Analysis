import re
from nltk.corpus import wordnet
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

import torchtext

def clean_data(df):
    drop_columns = ['published_date', 'published_platform', 'type', 'title', 'helpful_votes']

    df.drop(columns=drop_columns, inplace=True)
    
    # Title has 1 NaN value, need to fillna
    df = df.fillna("")

    df.to_csv('data_clean.csv', index=False)

def remove_emoji(text):
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


def tokenization(paragraphs, glove, lowercase=True, stop_words=True):
    '''
    Tokenize for 1 Paragraph
    '''
    tokenized_data = []
    for paragraph in paragraphs:
        if lowercase: paragraph = paragraph.lower()
        # Remove Emoji
        paragraph = remove_emoji(paragraph)

        tokens = paragraph.split()
        # Remove Punctuation
        paragraph = paragraph.translate(str.maketrans('', '', string.punctuation))
        # Remove Stop Words
        if stop_words:
            stops = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stops]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        
        tokens = [lemmatizer.lemmatize(word=token, pos=get_pos(tag)) for token, tag in pos_tag(tokens)]

        tokenized_paragraph = []
        for token in tokens:
            if token and token in glove.stoi:
                tokenized_paragraph.append(glove.stoi[token])
            else:
                tokenized_paragraph.append(glove.stoi['unk'])

        tokenized_data.append(tokenized_paragraph)

    return tokenized_data

