# Classification-Clustering Homework

This is the homework code for Education First using Python. For this exercise, it is asked to 
build a system to classify the Level of writing samples by English
language learners, using a data set gathered from users.

## Getting Started
Here is the structure of the code

EFWork/                
|
+--readme.md
| 
+--report.docx 
|  
+--classification/
|   |    
|   +--dataloading.py
|   +--preprocessing.py
|   +--featureextractor.py
|   +--plot.py
|   +--main.py
|   |
+--clustering/
|   |    
|   +--preprocessing.py
|   +--clusteringFunction.py
|   +--plot.py
|   +--main.py
|   |
|   |  
+--config/
|   +--logging.config
|   |  
+--data/
|   +--EFWritingData.xml
|   +--EFDataFrame.pk
|   +--EFDataFrame_sample=0.10.pk
|   |  
|   |  
+--figure/
|   |  

### Prerequisites

Make sure you have installed:
-python3.4 or higher
-pip
-nltk
---->nltk.download('punkt')
---->nltk.download('averaged_perceptron_tagger')
-pandas
-scikit-learn
-lxml



### Installing - Make it work
Where to start ?

For classification go to EFWork/classification/main.py
-------------------------------------------------------
In the main() function, you can choose which experiment to run.

Validation Experiment
- crossValidationExperimentWithXML() 
        Use this for cross-validation experiment given an xmlFile.
- crossValidationExperimentWithXML() 
        Use this for cross-validation experiment if you have already create the .pk file.
   
Test Experiment
- experimentWithXML() for train-test experiment 
    Use this for cross-validation experiment given an xmlFile.
- experimentWithDF() for train-test experiment 
    Use this for cross-validation experiment if you have already have a dataframe stored on a .pk file


The different parameters (common to all the experiments  functions)
    - classLabel : 'the class on which you want to perform the experiment.
        Possible values : level_id', 'group_id' 
    - classifierName : The name of different classes
        Possible values : 'NB'/'LR'/'DTree'/'KNN'/'SVM'
    - featureVecName = 'tfidfVect'/'countVect'/'tfidfNgramVect'/'customedFeaturesVect'
    - isSample = 
        Possible values : True/False
    - sampleSize : the percentage of data to keep in the sample
        Value : 0.1 for 10%
    -xmlFileName =  relative path to location of your xmlFile
        Given the organization of the projection used "../data/xmlfile.xml"
    -dfFileName : relative path to location of the stored dataframe
        Given the organization of the projection used "../data/dataframefile.pk"
    - cvType : 2 types, Cross validation can be performed to choose the right classifier or the right feature
        Possibile value : 'on-classifier' 'on-feature' 
    - cvFold = Number of fold for cross-validation


For clustering go to EFWork/clustering/main.py
-------------------------------------------------

You can run a Kmeans onf your data.
Specify the relative path to the location of '.pk' file where is stored your data and give the number of cluster you want to have.


##Logging info

The logging file "ef.log" is generated within you temporary folder

## Authors

**Tatiana Ekeinhor-Komi** 

## License

This project is licensed under the  BSD License - see the https://opensource.org/licenses/BSD-3-Clause  for details.


## Working

See details in the report.docx file  under EFWork/ folder
