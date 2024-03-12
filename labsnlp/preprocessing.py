import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(data):
    # removing signs, leaving only words
    print('Preprocessing text')
    print('Removing signs, leaving only words ')
    data['headline'] = data['headline'].str.lower(). \
                                                  str.replace(r'[^\w\s]+', '', regex=True). \
                                                  str.replace(r'\s+', ' ', regex=True).\
                                                  str.strip()

    # remove stopwords
    print('removing stopwords')
    stopwords_list = stopwords.words('english')
    data['headline'] = data['headline'].apply(lambda x: " ".join([word for word in x.split(' ') if word not in stopwords_list]))

    # lemmatize
    print('lemmatizing')
    lemmatizer = WordNetLemmatizer()
    data['headline'] = data['headline'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split(' ')]))

    return data