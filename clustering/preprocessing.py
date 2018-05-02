import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def tokenizing(text):
    """
    Return text tokenization.

    Parameters
        text : string
    Returns : list element
    """
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens


def loadData(fileName):
    """
    Return dataframe.

    Parameters
        text : string
    Returns : list element
    """
    try:
        dfAll = pd.read_pickle(fileName)
        df = dfAll[['level_id', 'group_id', 'group','topic', 'topic_id', 'level','text']].copy()
        return df
    except OSError as e:
        print("File not found at %s. Please check the location",fileName)

def loadReductedData(xLabel, yLabel, fileName = '../data/EFSampleDF.pk'):
    try:
        dfAll = pd.read_pickle(fileName)
        df = dfAll[[yLabel, xLabel]].copy()
        return df
    except OSError as e:
        print("File not found at %s. Please check the location",fileName)

def get_similarity_matrix(content_as_str):
    """
    returns similarity matrix
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=5000, min_df=0.2, stop_words='english',use_idf=True,
    tokenizer=tokenizing, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_as_str) #fit the vectorizer to synopses
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return (similarity_matrix, tfidf_matrix)