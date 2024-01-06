"""
Preprocessing
"""

import pandas as pd
import numpy as np
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist


# convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))



def main():
    df = pd.read_csv("IMDB Dataset.csv", nrows=20000)
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.head())

    df['text_clean'] = df['review'].apply(lambda x: finalpreprocess(x))
    print(df.head())
    '''
    dict_freq = {}
    for sentence in df['text_clean']:
        token = word_tokenize(sentence)
        for word in list(set(token)):
            if word in dict_freq:
                dict_freq[word] += 1
            else:
                dict_freq[word] = 1

    doc_num = len(df["text_clean"])
    for key, value in dict_freq.items():
        dict_freq[key] = [value, value/doc_num]

    tok_list = []
    for sentence in df['text_clean']:
        token = word_tokenize(sentence)
        new_tok_list = []
        for word in token:
            if dict_freq[word][1] < 0.20:
                new_tok_list.append(word)
            else:
                print(word)
        tok_list.append(new_tok_list)
    df["text_freq"]= [" ".join(i) for i in tok_list]
    '''


    df.to_csv('IMDB_clean.csv')








if __name__ == '__main__':
    main()


