import logging
import os.path
from lxml import etree
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def fast_iter(context, func):
    """
    Iterate through the xml tree and delete it after parsing and processing it.
        Parameters :
            context: string
            The path to the xmlFile
            func: callable
            the function that do the processing of the element of the xml file
        Returns :
            a pandas dataframe with the content of the xmlFile in context
    """

    logger.info("parsing - start...")
    dfcols = ['id','level', 'level_id', 'group', 'group_id','unit', 'topic', 'topic_id', 'grade', 'text']
    data = []
    for event, elem in context:
        oneElementLine = func(elem)
        if(len(oneElementLine[-1]) > 0):
            data.append(oneElementLine)
        # It's safe to call clear() here because no descendants will be accessed
        elem.clear()
        
        # Also eliminate now-empty references from the root node to elem
        for ancestor in elem.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]
    del context
    df_xml = pd.DataFrame(data, columns=dfcols)
    logger.info("parsing - end")
    return df_xml

def groupLevel(level):
    """
    Create the level to group mapping.
        Parameters :
            level: string
            the level to convert
        Returns :
            a string which represents the level to group mapping
    """
    if(level in ['1','2','3']):
        return 'A1',0
    elif(level in ['4','5','6']):
        return 'A2',1
    elif(level in ['7','8','9']):
        return 'B1',2
    elif(level in ['10','11','12']):
        return 'B2',3
    elif(level in ['13','14','15']):
        return 'C1',4
    else:
        return 'C2',5


def processElement(node):
    """
    Get a list of relevant information from an xml node
        Parameters :
            node: Xml Element
                The node in an XML Tree
        Returns :
            a list of extracted information
    """
    level = node.attrib.get('level')  #numerical variable are easier to learn
    levelid = int(level)
    writingId = node.attrib.get('id')
    unit = node.attrib.get('unit')
    topic = node.find('topic')
    topicid = topic.attrib.get('id')
    topictext = topic.text
    #date = node.find('date').text
    grade = node.find('grade').text
    #text = html.unescape(node.find('text').text)
    text = node.find('text').text.strip()
    group,groupid = groupLevel(level)

    return [writingId, 'L%02d'%(levelid+1), levelid ,group, groupid, int(unit), topictext,
    topicid, int(grade), text]
    

def loadXMLData(xmlFileName = "../data/EFWritingData.xml", dfFileName = "../data/EFDataFrame.pk"):
    """
        Convert the content of an xml file to a dataframe and store the result dataframe to a specified location.

        Parameters : 
            xmlFileName : string
                Relative path of the xmlFile.
            dfFileName : string
                Relative path of the file where the esult dataframe will be stored.
        Returns : None 
    """
    logger.info("processing - start...")

    currentFile = os.path.abspath(os.path.dirname(__file__))
    #xmlFileName = "../data/sampledata.xml"
    
    xmlFilePath = os.path.join(currentFile, xmlFileName)
    dfFilePath = os.path.join(currentFile, dfFileName)

    xmlWriting = etree.iterparse(xmlFilePath, events=('end', ), tag='writing' )
    xmlDataFrame = fast_iter(xmlWriting,processElement)

    xmlDataFrame.to_pickle(dfFilePath)
    logger.info("processing - end")

if __name__ == "__main__":
    loadXMLData()
    
