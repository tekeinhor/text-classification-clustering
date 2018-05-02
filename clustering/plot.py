import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def plotLegend(plt, topicDic):
    """
    Plot some information at the up left of the figure
    """
    ytext = 0.87 #3
    #for key, value in sorted(dict_example.items(), key=lambda x: x[0]): 
    for keyDic, topic in sorted(topicDic.items(), key = lambda x : int(x[0])):
        #stringTopic = "%s(L-%s) : %s\n"%(keyDic,str(topicLevelDic[keyDic]),topic)
        stringTopic = "(L-%s) : %s\n"%(keyDic,topic)
        #ax.text(4, ytext, stringTopic, style='italic') #bbox={'facecolor':'None', 'alpha':0.5, 'pad':10}
        plt.text(0.8, ytext, stringTopic, fontsize=9, transform=plt.gcf().transFigure)
        ytext = ytext - 0.02 #-0.15
    
    plt.subplots_adjust(right=0.8)


def scatter_clusters(x_pos, y_pos, clusters, titles, figName,topicDic = {}, topicLevelDic = {}):
    """
    plot the document on figure.
    """
    cluster_colors = {0: '#cc0000', #red
                      1: '#006600', #green
                      2: '#002699', #blue
                      3: '#ffff33', #yellow
                      4: '#ffa64d', #orange
                      5: '#000000', #black
                      6: '#a84dff', #magenta
                      7: '#ee42f4' #pink
                      }
    # As many as items
    cluster_names = {0: 'Group1',
                1: 'Group2',  
                2: 'Group3', 
                3: 'Group4',
                4: 'Group5',
                5: 'Group6',
                6: '#Group7', 
                7: '#Group8' }
    

    df = pd.DataFrame(dict(x= x_pos, y= y_pos, label= clusters, title= titles)) 
    groups = df.groupby('label')
    
    fig, ax = plt.subplots(figsize=(10, 9))  # Set size
    ax.set_facecolor('#e6f7ff') #very light blue

    # Iterate through groups to layer the plot
    for name, group in groups:
        line, = ax.plot(group.x, group.y, marker='D', alpha = 0.2, linestyle='None', ms=15, 
                label=cluster_names[name], color=cluster_colors[name], mec='black')
        ax.set_aspect('auto')
        ax.tick_params(axis= 'x', which='both', labelbottom='off')
        ax.tick_params(axis= 'y', which='both', labelleft='off')
    ax.legend(numpoints=1)
    
    
    
    #for i in range(len(df)):
    #    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size= 15) #
    
    #dateTime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    now = datetime.now()
    dateTime = now.strftime("%Y-%m-%d %H:%M")
    
   
    #plt.show() # Show the plot

    plt.savefig(figName)