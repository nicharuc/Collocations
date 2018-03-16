import pandas as pd
import numpy as np
import spacy
import re
import string

#load reviews data
reviews = pd.read_csv('7282_1.csv')
#extract only reviews
comments = reviews['reviews.text']
comments = comments.astype('str')

#function to remove non-ascii characters
def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)
#remove non-ascii characters
comments = comments.map(lambda x: _removeNonAscii(x))

#get stop words of all languages
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
#function to detect language based on # of stop words for particular language
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    lang = max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
    if lang == 'english':
        return True
    else:
        return False
#filter for only english comments
eng_comments=comments[comments.apply(get_language)]

#drop duplicates
eng_comments.drop_duplicates(inplace=True)

#load spacy
nlp = spacy.load('en')

#function to clean and lemmatize comments
def clean_comments(text):
    #remove punctuations
    regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
    nopunct = regex.sub(" ", str(text))
    #use spacy to lemmatize comments
    doc = nlp(nopunct, disable=['parser','ner'])
    lemma = [token.lemma_ for token in doc]
    return lemma

#apply function to clean and lemmatize comments
lemmatized = eng_comments.map(clean_comments)

#make sure to lowercase everything
lemmatized = lemmatized.map(lambda x: [word.lower() for word in x])

#turn all comments' tokens into one single list
unlist_comments = [item for items in lemmatized for item in items]
