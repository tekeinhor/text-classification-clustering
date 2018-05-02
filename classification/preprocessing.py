import os.path
import pandas as pd
import re
import sys
from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from nltk.tag import pos_tag
import nltk
import logging
from featureextractor import MyExtractor
from sklearn.pipeline import Pipeline, FeatureUnion

logger = logging.getLogger(__name__)


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


def loadData(xLabel, yLabel, fileName = '../data/EFSampleDF.pk'):
    """
    load a specific part of a (previously stored) data at given file.
    """
    try:
        dfAll = pd.read_pickle(fileName)
        df = dfAll[[yLabel, xLabel]].copy()
        return df
    except OSError as e:
        logger.error("load data -File not found at %s -Please check the location",fileName)
    

def loadDataAll(fileName = '../data/EFSampleDF.pk'):
    """
    load the whole (previously stored) data at given file.
    """
    try:
        return pd.read_pickle(fileName)
        
    except OSError as e:
        logger.error("loadAll -File not found at %s -Please check the location",fileName)
        #sys.exit(0)

def complexSample(df, label, size):
    """
    Returns a the result dataframe after performing sampling.
    Keep the distribution of the given index (here our classs label) of the dataframe.

    Parameters:
        df : pd.dataframe
        label : string
        size : float
    Returns : pd.dataframe
    """
    grouped = df.groupby(label)
    frames = []
    for k,v in grouped:
        sampleV = v.sample(frac=size)
        #print('level %s from  %d to %d'%(k, v.shape[0], sampleV.shape[0]))
        frames.append(sampleV)
    return  pd.concat(frames).reset_index(drop=True)

def simpleSample(df, size):
    """
    Returns a the result dataframe after performing sampling.
    
    Parameters:
        df : pd.dataframe
        label : string
        size : float
    Returns : pd.dataframe
    """
    return df.sample(frac=size)
 # first tokenize by sentence
def tokenizing(text):
    """
    Returns text tokenization.

    Parameters
        text : string
    Returns : list element
    """
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    return tokens


# 
def checkPOSTag(text, flag):
    """
    Returns the part of speech tag count of a words in a given sentence.
    Parameters
        text : string
        flag: string
            Part of Speech to count
    Returns : int
    """
    cnt = 0
    x = tokenizing(text)
    tagged = pos_tag(x)
    for tup in tagged:
        ppo = list(tup)[1]
        if ppo in pos_family[flag]:
            cnt += 1
  
    return cnt


def createCountVect(document,xTrainSet):
    """
    Create a count vectorizer object.

    Parameters
        document : dataframe
            dataframe representing the writings
        xtrainSet = array
    Returns : the result features and the vectorizer
    """
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(document)

    # transform the training  data using count vectorizer object
    count =  count_vect.transform(xTrainSet)

    return count,count_vect

def createTfidfVect(document,xTrainSet):
    """
    Create a word level tf-idf object

    Parameters
        document : dataframe
            dataframe representing the writings
        xtrainSet = array
    Returns : the result features and the vectorizer
    """
    
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(document)
    tfidf =  tfidf_vect.transform(xTrainSet)
    
    return tfidf,tfidf_vect

def createTfidfNgramVect(document, xTrainSet):
    """
    Create a ngram level tf-idf

    Parameters
        document : dataframe
            dataframe representing the writings
        xtrainSet = array
    Returns : the result features and the vectorizer
    """
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(document)
    tfidf_ngram =  tfidf_vect_ngram.transform(xTrainSet)
     #tfidf_vect.fit(df['text'])

    return tfidf_ngram,tfidf_vect_ngram

def createTfidfCharVect(document, xTrainSet):
    """
    Create a characters level tf-idf object

    Parameters
        document : dataframe
            dataframe representing the writings
        xtrainSet = array
    Returns : the result features and the vectorizer
    """
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(document)
    tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(xTrainSet) 

    return tfidf_ngram_chars,tfidf_vect_ngram_chars

def createCombinedExtractor():
    """
    Create a combination of feature vect.
    Returns : the combination of different features vect
    """
    featureFuncList = [getAvgWordPerSentence, getSentenceCount , getLongestSentence, getPunctuationCount, getAdvCount, getAdjCount]
    featureFuncNameList = ['AvgWordPerSentence', 'SentenceCount' , 'LongestSentence', 'PunctuationCount', 'AdvCount', 'AdjCount']
    #tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    estimators = [ (name,MyExtractor(func)) for name,func in zip(featureFuncNameList, featureFuncList)]
    #estimators.extend(('tf-idf', tfidf_vect)))
    return FeatureUnion(estimators)
  
def createFeaturesVect(document, xTrainSet):
    """ 
    Create a customed features vect object.

    Parameters
        document : dataframe
            dataframe representing the writings
        xtrainSet = array
    Returns : the result features and the vectorizer
    """
    combined_extractorVect = createCombinedExtractor()
    combined_extractorVect.fit(document)
    combined_features = combined_extractorVect.transform(xTrainSet)
    return combined_features, combined_extractorVect


sentenceCount  = lambda x: [len(l.split()) for l in re.split(r'[?!.]', ''.join(x)) if l.strip()]

def getAvgWordPerSentence(text):
    """
    Returns the average number of words in a sentence in a text.
    Example : 'My Name is Jane. I am a twenty years old', result 5.
    """
    sentencelist = sentenceCount(text)
    if(len(sentencelist) == 0):
        return 0
    else:
        return sum(sentencelist)/len(sentencelist)

def getSentenceCount(text):
    """
    Return the number of sentences in a text.
    Example : 'My Name is Jane. I am a twenty years old', result 2.
    """
    return len(sentenceCount(text))

def getLongestSentence(text):
    """
    Returns the lenght of longest sentence in a text.
    Example : 'My Name is Jane. I am a twenty years old', result 6.
    """
    count = sentenceCount(text)
    if(len(count) >0):
        return max(count) 
    else:
        return 0
    
def getPunctuationCount(text):
    """
    Returns the number of punctuation in a text. List of punctuation : (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~).
    Example : 'My Name is Jane. I am a twenty years old', result 2.
    """
    return len("".join(_ for _ in text if _ in string.punctuation))

def getPOSCount(text, pos):
    
    return checkPOSTag(text, pos)

def getAdvCount(text):
    """
    Returns the count of adverb in a given sentence.
    Parameters
        text : string
        flag: string
            Part of Speech to count
    Returns : int
    """
    return checkPOSTag(text, 'adv')

def getAdjCount(text):
    """
    Returns the count of adverb in a given sentence.
    Parameters
        text : string
        flag: string
            Part of Speech to count
    Returns : int
    """
    return checkPOSTag(text, 'adj')

def createNaturalFeatures(df):
    
    countPunctuation = lambda x: len("".join(_ for _ in x if _ in string.punctuation))
    sentenceLength  = lambda x: [len(l.split()) for l in re.split(r'[?!.]', ''.join(x)) if l.strip()]
    upperCaseCount = lambda x: len([wrd for wrd in x.split() if wrd.isupper()])

    #df['char_count'] = df['text'].apply(len)
    #df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    #df['word_density'] = df['char_count'] / (df['word_count']+1)
    #df['punctuation_count'] = df['text'].apply(countPunctuation) 
    #df['word_count_per_sentence_vec'] = df['text'].apply(sentenceLength)
    #df['sentence_count'] = df['word_count_per_sentence_vec'].apply(len)
    #df['min_sentence'] = df['word_count_per_sentence_vec'].apply(min) 
    #df['max_sentence'] = df['word_count_per_sentence_vec'].apply(max)
    #df['word_count_per_sentence_avg'] = df['word_count_per_sentence_vec'].apply(sum) / df['sentence_count']
    #df['upper_case_word_count'] = df['text'].apply(upperCaseCount)

    #df['adv_count'] = df['text'].apply(lambda x: checkPOSTag(x, 'adv'))
    #df['adj_count'] = df['text'].apply(lambda x: checkPOSTag(x, 'adj'))



def main():
    currentFile = os.path.abspath(os.path.dirname(__file__))
    dfFileName = "../data/EFSampleDF.pk"
    dfFilePath = os.path.join(currentFile, dfFileName)
    xLabel = 'text'
    yLabel = 'level_id'
    efdata = loadData(xLabel, yLabel, dfFilePath)

    trainX, testX, trainY, testY = train_test_split(efdata[xLabel], efdata[yLabel],random_state=0, test_size=0.2)

    print('------\n',trainX[:5])
    print('------\n',trainY[:5])
    print('------\n',testX[:5])
    print('------\n',testY[:5])

    print(efdata[xLabel])

    count, countVect = createTfidfVect(efdata[xLabel],trainX)

    print(count)
if __name__ == "__main__":

    text = "After some time, the affection between them is progressing well !\
    John's personality deeply moved Isabella.\
    So Isabella decided to break up with Tom and fell in love with John ?"
 
