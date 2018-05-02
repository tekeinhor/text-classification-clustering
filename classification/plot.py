import os.path
import matplotlib.pyplot as plt
from preprocessing import loadDataAll, createNaturalFeatures
currentFile = os.path.abspath(os.path.dirname(__file__))
def plotting(xLabel, yLabel, x, y, figName = '', save = False):
    currentFile = os.path.abspath(os.path.dirname(__file__))
    figPath = os.path.join(currentFile, figName)

    fig, ax = plt.subplots()
    ax.scatter(x, y, edgecolors=(0, 0, 0))
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', lw=4)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)   
    
    if(save):
        fig.savefig(figPath)
    else:
        plt.show()


def plotCountBarChart(x, label, figName):
    """
    Plot the bar chart of a given data and store the figure.
    Parameters :
        x : dataframe
            A dataframe containing the data to plot
        label : string
            It represents the class used for the data
    Returns: None
    """
    fig = plt.figure(figsize=(8,6))
    x.plot.bar(ylim=0)
    
    plt.title('Bar Chart : Distribution of writings per %s'%label)
    plt.ylabel('Number of Writings')
    plt.xlabel('Class')
    
    #if('level' in label):
    #    numberOfClass = len(x)
    #    plt.xticks(list(range(numberOfClass)),['L%d'%(i+1) for i in range(numberOfClass)])

    #plt.show()
    print('saving...')
    plt.savefig(figName)

def plotErrorBar(mean, std, xLabel, yLabel, figName):
    """
    Plot the error bar given mean and std and store the obtained figure.
    Returns: None
    """
    fig = plt.figure(figsize=(8,6))
    plt.errorbar(range(len(mean.index)), mean, yerr=std, fmt = 'o', linestyle='', capsize = 10 ,ecolor = 'r')
   
    plt.title('Error Chart : Mean of %s per %s'%(yLabel, xLabel))
    plt.ylabel(yLabel)
    plt.xlabel('Class %s'%xLabel)
   
    plt.xticks(range(len(mean.index)),mean.index)
    plt.savefig(figName)
  

def countBarchat(df, category, isSample):
    """
    Plot the Bar chart given a dataframe and some class information then store the obtained figure.
    Returns: None
    """
    ######Plot  Bar#########
    h = df.groupby(category).text.count()
    figName = '../figure/barchart_writingdistribution_%s_issample%r.pdf'%(category,isSample)
    figFilePath = os.path.join(currentFile, figName)
    plotCountBarChart(h,category,figFilePath)

def errorBar(df, category, feature,isSample):
    """
    Plot the Bar chart given a dataframe grouped by a category (class of data) and some given features and create a figure.
    Paramters:
        df : dataframe
            The dataframe containing the data to plot.
        category: string
            The classification label of the data
        feature: string
            feature for the groupby
        isSample: bool
            info used for the figure name.
    Returns: None
    """
    
    #feature = "sentence_count"  
    mean = df.groupby(category)[feature].mean()
    std = df.groupby(category)[feature].std()
   
    figName = '../figure/errorbar_%s_%s_issample=%r.pdf'%(feature,category,isSample)
    figFilePath = os.path.join(currentFile, figName)

    plotErrorBar(mean, std, category, feature, figFilePath)


def datavisualization():
    #dfFileName = "../data/EFSampleDF.pk"
    dfFileName = "../data/EFDataFrame.pk"
    dfFilePath = os.path.join(currentFile, dfFileName)
    print("load data...")
    df = loadDataAll(dfFilePath)

    print('featureCreation...')
    createNaturalFeatures(df)

    if 'sample' in dfFileName.lower():
        isSample =True
    else:
        isSample = False
    #######Error bar#########
    #________
    """print("plot 1.1...")
    countBarchat(df,'level', isSample)
    print("plot 1.1 - end")
    print("plot 1.2...")
    errorBar(df,'level','grade', isSample)
    print("plot 1.2 - end")
    print("plot 1.3...")
    errorBar(df,'level','word_count_per_sentence_avg', isSample) 
    print("plot 1.3 - end") """
    
    print("plot 2.1...")
    countBarchat(df,'group', isSample)
    print("plot 2.1 - end")
    print("plot 2.2...")
    errorBar(df,'group','grade', isSample)
    print("plot 2.2 - end")
    print("plot 2.3...")
    errorBar(df,'group','word_count_per_sentence_avg',isSample)
    print("plot 2.3 - end")

    print('end')


if __name__ == "__main__":

    datavisualization()
    

