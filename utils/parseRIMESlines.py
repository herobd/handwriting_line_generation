import sys
import xml.etree.ElementTree as ET 
from xml.sax.saxutils import unescape as unescape_
import json
from collections import defaultdict
#import imageio

def unescape(s):
    return unescape_(s).replace('&quot;','"').replace('&apos;',"'")


def getLineBoundaries(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    pageLines=defaultdict(list)
    for page in root.findall('SinglePage'):
        image = page.attrib['FileName']
        image = image[image.index('/')+1:]
        allHs=0
        lines=[]
        for line in page.findall('Paragraph/Line'):

            trans=unescape(line.attrib['Value'])
            top = int(line.attrib['Top'])
            bot = int(line.attrib['Bottom'])
            left = int(line.attrib['Left'])
            right = int(line.attrib['Right'])
            lines.append(([top,bot+1,left,right+1],trans))
            allHs+=1+bot-top
        meanH = allHs/len(lines)
        for bounds,trans in lines:
            diff = meanH-(bounds[1]-bounds[0])
            if diff>0:
                #pad out to make short words the same height on the page
                bounds[0]-=diff/2
                bounds[1]+=diff/2
            #but don't clip in tall words

            #add a little extra padding horizontally
            bounds[2]-= meanH/4
            bounds[3]+= meanH/4
            bounds = [round(v) for v in bounds]
            #lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            pageLines[image].append((image,bounds,trans))
    return pageLines
