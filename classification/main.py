import os.path
import numpy as np
from datetime import datetime
import tempfile
import logging
import logging.config
from preprocessing import createFeaturesVect, createTfidfVect,createTfidfNgramVect, loadData, loadDataAll, createCountVect, complexSample
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import linear_model, naive_bayes, svm, tree, neighbors
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from dataloading import loadXMLData
import sys

currentFileDir = os.path.abspath(os.path.dirname(__file__))
logFile = os.path.join(tempfile.gettempdir(), "ef.log")
loggerConfigPath = os.path.join(currentFileDir, "../config/logger.config")
logging.config.fileConfig(loggerConfigPath, defaults={'logfilename': logFile},disable_existing_loggers = False)
logger = logging.getLogger(__name__) #logger object creation

classifiers = {'LR' :  linear_model.LogisticRegression(), 'KNN': neighbors.KNeighborsClassifier(),
            'DTree': tree.DecisionTreeClassifier(), 'NB': naive_bayes.MultinomialNB(), 'SVM': svm.SVC() }

createFeaturesVec = {'tfidfVect' : createTfidfVect, 'countVect' : createCountVect ,
            'tfidfNgramVect' : createTfidfNgramVect, 'customedFeaturesVect' : createFeaturesVect}

def compareFeatures(classifierName, efdata, trainX, trainY, testX, cvFold):
    """
        For a given classifier, compute cross validation on train data using different settings of features.
        And displa the confusion matrix info and the reporting (precision, recall, fmeasure and support)
        Parameters:
            classifierName: string
                the name of classifier to use
            efdata: pd.dataframe
                the dataset
            trainX: pd.dataframe (subsample of efdata)
                the train data set for X
            trainY: pd.dataframe (subsample of efdata)
                the train data set for Y
            testX: pd.dataframe (subsample of efdata)
                the test data set for Y
            cvFold : int
                number of fold for the cross validation
        Returns: None        
    """
    #featureFuncList = [createTfidfVect, createCountVect , createTfidfNgramVect, createFeaturesVect]
    #featureFuncNameList = ['createTfidfVect', 'createCountVect' , 'createTfidfNgramVect', 'createFeaturesVect']
    classifier = classifiers[classifierName]
    
    logger.info('Classifier used : %s',classifierName)
    for name,func in createFeaturesVec.items():
        logger.info("-----------Feature %s-----------",(name))
        xtrain_count, featureVec = func(efdata['text'],trainX)
        #sxtest_count = featureVec.transform(testX)
        y_pred = cross_val_predict(classifier, xtrain_count, trainY ,cv=cvFold)
        conf_mat = confusion_matrix(trainY,y_pred) #conf_mat
        report = classification_report(trainY, y_pred)
        logger.debug("Confusion Matrix  ===> \n%s",conf_mat)
        logger.info("Reporting \n%s",report)

    
def compareAlgo(feature_vector_train, Y_train, cvFold, scoring, figName):
    """
        For a given (training) dataset, compute Kfold cross-validation with different classifiers.
        Differents measures are used for the performance measurement. (accuracy, confusion matrix info and the reporting (precision, recall, fmeasure and support).
        
        Parameters:
            classifierName: string
                the name of classifier to use.
            feature_vector_train: array
                the features representing the train dataset.
            Y_train: pd.dataframe (subsample of efdata)
                the train data set for Y.
            cvFold : int
                number of fold for the cross validation.
            scoring: string
                scoring method to use.
            figName: string
                Path to file where the figure of the classifiers accucracy is stored, when using the 'accuracy' scoring.
        
        Returns: None        
    """
    results = []
    names = []
    
    for name, classifier in classifiers.items():
        logger.info('---------- %s ------------',name)
        #cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        if(scoring == 'accuracy'):
            cv_results = cross_val_score(classifier, feature_vector_train, Y_train, cv=cvFold)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            logger.info("CV RESULT  ===> %r",cv_results)
            logger.info(msg)
            plotAlgo(results,names, figName)
        else:
            y_pred = cross_val_predict(classifier, feature_vector_train, Y_train ,cv=cvFold)
            conf_mat = confusion_matrix(Y_train,y_pred) #conf_mat
            report = classification_report(Y_train, y_pred)
            logger.debug("Confusion Matrix  ===> \n%r",conf_mat)
            logger.info("Reporting\n %s",report)
       


def plotAlgo(results, names, figName):
    """
        Plot a boxplot for a given setting.
        Returns: None        
    """
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy')
    #plt.yticks(np.arange(0, 1, 0.1)) 
    plt.savefig(figName)


def sampling(sampleSize, dfFilePath, dfSampleFileName):
    """
    Perform a sampling on a dataset at the given path keeping the same distribution per class category (level) as in the big dataset.
        
        Parameters:
            sampleSize: float (default : 10%)
                the percentage of data to use in the sampling
            dFileName: string
                path to data
        Returns: pd.dataframe
            result sampling
           
    """
    allData = loadDataAll(dfFilePath)
    if(sampleSize == 1):
        efdata =  allData
    else:
        efdata = complexSample(allData, 'level_id', sampleSize)
    #dfSampleFileName = "../data/EFSampleDF.pk"
    dfSampleFilePath = os.path.join(currentFileDir, dfSampleFileName)

    efdata.to_pickle(dfSampleFilePath)
    return efdata


def testAlgo(classifierName, feature_vector_train, Y_train, feature_vector_test, Y_test, figFilePath):
    """
        The result of the classifcation performed on a test set after training on the given training test.
        Differents measures are used for the performance measurement. (confusion matrix info and the reporting (precision, recall, fmeasure and support).
        Parameters:
            classifierName: string
                the name of classifier to use.
            feature_vector_train: array|ndarray|sparseMatrix
                the features representing the train dataset.
            Y_train: pd.dataframe
                the train data set for class Y.
            feature_vector_test: array|ndarray|sparseMatrix
                the features representing the test dataset.
            Y_test: pd.dataframe
                the train data set for class Y.
            figFilePath: string
                Path to file where a figure will be saved.
        Returns: None        
    """
    logger.info('Classifier %s',classifierName)   
    classifier = classifiers[classifierName]
    classifier.fit(feature_vector_train, Y_train)
    y_pred = classifier.predict(feature_vector_test)
    conf_mat = confusion_matrix(Y_test,y_pred) #conf_mat
    report = classification_report(Y_test, y_pred)
    logger.debug("Confusion Matrix  ===> \n%s",conf_mat)
    logger.debug("Reporting\n %s",report)


def main():
    classLabel = 'level_id'
    classifierName = 'NB'
    featureVecName = 'tfidfVect'
    isSample = True
    sampleSize =  0.1

    cvType = 'on-feature' #on-classifier
    cvFold = 2
    xmlFileName = "../data/sampledata.xml" # "../data/EFWritingData.xml"
    dfFileName = "../data/sampledata.pk" #EFDataFrame

    #experimentWithXML(classLabel, classifierName, featureVecName, isSample, sampleSize, xmlFileName, dfFileName)
    #experimentWithDF(classLabel, classifierName, featureVecName, isSample, sampleSize, dfFileName)
    #crossValidationExperimentWithXML(classLabel, isSample, sampleSize, cvType, cvFold, classifierName,featureVecName, xmlFileName, dfFileName)
    crossValidationExperimentWithDF(classLabel, isSample, sampleSize, cvType, cvFold, classifierName,featureVecName, dfFileName)
    experimentWithDF(classLabel, classifierName, featureVecName, isSample, sampleSize, dfFileName)
    

def experimentWithXML(classLabel, classifierName, featureVecName, isSample, sampleSize, xmlFileName = "../data/EFWritingData.xml", dfFileName = "../data/EFDataFrame.pk"):     
    """
        Perfom classification experiment for a given xmlFile representing the dataset (to classify).
        
        Parameters :
            classLabel : string
                The class Label, group_id or level_id
            classifierName : string
                Name of the classifier to use for the classifier : (naive bayes, logistic regression, knn ...)
            featureVecName : string
                The type of features Vector to use, tfidfVect, countVec, or customedfeatureVec.
            isSample : boolean
                Indicate if the subsample of the dataset should be used for the classification
            sampleSize : float
                size of the sample dataset if it applies
            xmlFileName: string
                relative path to the xmlFile (representing the dataset) to parse.
            dFileName : string
                relative path to the location of stored dataframe, representing the whole dataset.
        Returns : None
    """
    logging.info('Start processing - xml')
    #Load the data at the given fileName location for the given
    loadXMLData(xmlFileName, dfFileName)

    #Experiment With dataframe already stored
    experimentWithDF(classLabel, classifierName, featureVecName, isSample, sampleSize, dfFileName)
    logger.info('END processing - xml')
    
def renameFileName(fileName, toAdd):
    """ 
        rename a fileName. Modify the basename of a path string with a given string.
        Example modify 'data.pk' to 'data_sample.pk'.

        Parameter :
            fileName : string
                relative path to fileName
            toAdd : 
                string to add to the original fileName
        Returns : a new file Name (string) 

    """
    baseName, ext = os.path.splitext(fileName)
    newName = '%s_%s%s'%(baseName,toAdd,ext)
    return newName

def experimentWithDF(classLabel, classifierName, featureVecName, isSample, sampleSize = 0.1, dfFileName = "../data/EFDataFrame.pk"):
    """
        Perfom classification experiment for a given dataframe representing the dataset (to classify).
        
        Parameters :
            classLabel : string
                The class Label, group_id or level_id
            classifierName : string
                Name of the classifier to use for the classifier : (naive bayes, logistic regression, knn ...)
            featureVecName : string
                The type of features Vector to use, tfidfVect, countVec, or customedfeatureVec.
            isSample : boolean
                Indicate if the subsample of the dataset should be used for the classification
            sampleSize : float
                size of the sample dataset if it applies
            dFileName : string
                relative Path to the stored dataframe representing the whole dataset.
        Returns : None
    """
    logger.info("Start: Logger file info here : %r",logFile)
    figName = '../figure/experiment_%s_isSample=%r.pdf'%(classLabel,isSample)
    figFilePath = os.path.join(currentFileDir, figName)
    logger.info('%s - load data...',classLabel)

    #Sampling
    logger.info('Data Sampling - %.2f percent of data',sampleSize*100)
    if(isSample):
        dfFilePath = os.path.join(currentFileDir, dfFileName)

        sampleDfFileName = renameFileName(dfFileName, 'sample=%.2f'%sampleSize)
        sampleDfFilePath = os.path.join(currentFileDir, sampleDfFileName)
        efdata = sampling(sampleSize, dfFilePath, sampleDfFilePath)
        logger.info('Sampled Data file is at the location - %s',sampleDfFilePath)

    else:
        dfFilePath = os.path.join(currentFileDir, dfFileName)
        efdata = loadData('text', classLabel, dfFilePath)

    logger.info('Number of writings in working data : %r',efdata.shape[0])

    #Train -Test split
    logger.info('train-test split...')
    xtrain_df, xtest_df, ytrain_df, ytest_df = train_test_split(efdata['text'], efdata[classLabel], random_state=0, test_size=0.2)

    #feature
    logger.info('features computation with %s ...',featureVecName)
    featureVecFunction = createFeaturesVec[featureVecName]
    xtrain_vec, featureVec =   featureVecFunction(efdata['text'],xtrain_df)
    xtest_vec = featureVec.transform(xtest_df)

    #Test Algo
    logger.info('Test\n___________________________')
    testAlgo(classifierName, xtrain_vec, ytrain_df, xtest_vec, ytest_df, figFilePath)

    logger.info('END processing on dataframe')

def crossValidationExperimentWithXML(classLabel, isSample, sampleSize, cvType, cvFold, classifierName,featureVecName, xmlFileName, dfFileName):
    """
    Perfom cross-validation for a classification experiment, given an XML File representing the dataset (to classify).
    Two option are provided. Perform cross validation to find the right features and/or the right classifier.
    
    Parameters :    
        classLabel : string
            The class Label, group_id or level_id
        isSample : boolean
            Indicate if the subsample of the dataset should be used for the classification
        sampleSize : float
            size of the sample dataset if it applies
        cvType : string
            One can perform cross-validation to feature selection development of for classification algorithm development.
        cvFold: int
            Number of folder for cross-validation
       classifierName : string
            Name of the classifier to use for the classifier : (naive bayes, logistic regression, knn ...)
            If perfoming 'on-feature' type of cross-validation, one needs to provide a classifier to cross-validate on.
        featureVecName : string
            The type of features Vector to use, tfidfVect, countVec, or customedfeatureVec.
            If perfoming 'on-classifier' type of cross-validation, one needs to provide a type of featureVec to cross-validate on.
        xmlFileName : string
            Relative path to the xmfFile containing the dataSet to perform on.
        dFileName : string
            relative Path to the stored dataframe representing the whole dataset.
    Returns : None
    """
    logger.info('Start Processing - xml')
    #Load the data at the given fileName location for the given
    loadXMLData(xmlFileName, dfFileName)

    #Experiment With dataframe already stored
    crossValidationExperimentWithDF(classLabel, isSample, sampleSize, cvType, cvFold, classifierName,featureVecName, dfFileName)
    logger.info('End Processing - xml')
def crossValidationExperimentWithDF(classLabel, isSample, sampleSize, cvType, cvFold, classifierName, featureVecName, dfFileName = "../data/EFDataFrame.pk"):
    """
    Perfom cross-validation for a classification experiment, given dataframe representing the dataset (to classify).
    Two option are provided. Perform cross validation to find the right features and/or the right classifier.
    
    Parameters :
        classLabel : string
            The class Label, group_id or level_id
        isSample : boolean
            Indicate if the subsample of the dataset should be used for the classification
        sampleSize : float
            size of the sample dataset if it applies
        cvType : string
            One can perform cross-validation to feature selection development of for classification algorithm development.
        cvFold: int
            Number of folder for cross-validation
       classifierName : string
            Name of the classifier to use for the classifier : (naive bayes, logistic regression, knn ...)
            If perfoming 'on-feature' type of cross-validation, one needs to provide a classifier to cross-validate on.
        featureVecName : string
            The type of features Vector to use, tfidfVect, countVec, or customedfeatureVec.
            If perfoming 'on-classifier' type of cross-validation, one needs to provide a type of featureVec to cross-validate on.
        dFileName : string
            relative Path to the stored dataframe representing the whole dataset.
    Returns : 
        None
    """

    figName = '../figure/%d-FolcvExperiment_%s_%s_isSample=%r.pdf'%(cvFold,cvType,classLabel,isSample)
    figFilePath = os.path.join(currentFileDir, figName)
    logger.info('%s - load data...',classLabel)
    #Sampling
    logger.info('Data Sampling - %.2f percent of data',sampleSize*100)
    if(isSample):
        dfFilePath = os.path.join(currentFileDir, dfFileName)
        sampleDfFileName = renameFileName(dfFileName, 'sample=%.2f'%sampleSize)
        sampleDfFilePath = os.path.join(currentFileDir, sampleDfFileName)
        efdata = sampling(sampleSize, dfFilePath, sampleDfFilePath)
        logger.info('Sampled Data file is at the location - %s',sampleDfFilePath)

    else:
        dfFilePath = os.path.join(currentFileDir, dfFileName)
        efdata = loadData('text', classLabel, dfFilePath)

    logger.info('Number of writings in working data : %r',efdata.shape[0])

    #Train -Test split
    logger.info('Train-test split : 80-20')
    xtrain_df, xtest_df, ytrain_df, ytest_df = train_test_split(efdata['text'], efdata[classLabel], random_state=0,test_size=0.2)


    #feature
    logger.info('Feature - %s ',featureVecName)
    featureVecFunction = createFeaturesVec[featureVecName]
    xtrain_vec, featureVec =   featureVecFunction(efdata['text'],xtrain_df)
    xtest_vec = featureVec.transform(xtest_df)

   
    logger.info('Cross Validation - %s ...',cvType)
    if(cvType == 'on-feature'):
        #Cross Validation - Features Selection
        logger.info('Comparing Different Feature-Vectors')
        compareFeatures(classifierName, efdata, xtrain_df, ytrain_df, xtest_df, cvFold)
    else:
        #Cross Validation - Algo Selection
        logger.info('Comparing Different Classifiers')
        compareAlgo(xtrain_vec,ytrain_df, cvFold, scoring, figFilePath)

    logger.info('End-Processing')

if __name__ == "__main__":
    #dataLoading
    """sampleSize = 0.01
    dfFileName = '../data/EFDataFrame.pk'
    dfFilePath = os.path.join(currentFileDir, dfFileName)
    sampleDfFileName = renameFileName(dfFileName, 'sample=%.2f'%sampleSize)
    sampleDfFilePath = os.path.join(currentFileDir, sampleDfFileName)
    efdata = sampling(sampleSize, dfFilePath, sampleDfFilePath)
    #print(renameFileName('/data/file.txt', 'sample'))"""

    main()
