import os.path
from preprocessing import loadData,get_similarity_matrix
from clusteringFunction import get_cluster_kmeans, pca_reduction
from plot import scatter_clusters
import logging, logging.config
import tempfile

currentFileDir = os.path.abspath(os.path.dirname(__file__))
logFile = os.path.join(tempfile.gettempdir(), "ef.log")
loggerConfigPath = os.path.join(currentFileDir, "../config/logger.config")
logging.config.fileConfig(loggerConfigPath, defaults={'logfilename': logFile},disable_existing_loggers = False)
logger = logging.getLogger(__name__) #logger object creation



def main(num_clusters = 6, dfFileName = "../data/EFDataFrame_sample=0.01.pk"):
    logger.info('Start')

    currentFile = os.path.abspath(os.path.dirname(__file__))
    dfFilePath = os.path.join(currentFile, dfFileName)
    label = 'group'

    #Load the data at the given fileName location for the given
    logger.info('load data')
    efdata = loadData(dfFilePath)
    logger.info('Number of element %d',efdata.shape[0])

    #Extract Features
    logger.info('extract features')
    (similarity_matrix, tfidf_matrix) = get_similarity_matrix(efdata['text'])
    #_____________________________________________________________________
    #______________________________KMEANS_________________________________
    #peform KMEANS
    
    logger.info('------ K-means : %d--------',num_clusters)
    titles = efdata[label] 
    km_clusters = get_cluster_kmeans(tfidf_matrix, num_clusters, titles)
    #
    logger.info('------ Dimensions reduction --------') 
    x_pos, y_pos = pca_reduction(similarity_matrix, 10)

   
    #res = efdata.set_index('topic_id')['topic'].to_dict()
    #res2 = efdata.set_index('level')['topic'].to_dict()
    logger.info('plot')
    figName = '../figure/clustering_experiment_%s_isSample=True.pdf'%(label)
    figFilePath = os.path.join(currentFileDir, figName)
    scatter_clusters(x_pos, y_pos, km_clusters, titles, figFilePath) # Scatter K-means with PCA
    logger.info('End')
    
if __name__ == "__main__":
    dfFileName = "../data/EFDataFrame_sample=0.01.pk"
    num_clusters = 6
    main(num_clusters, dfFileName)
   
