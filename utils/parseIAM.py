import sys
import xml.etree.ElementTree as ET 
from xml.sax.saxutils import unescape as unescape_
import json
from collections import defaultdict
#import imageio

def unescape(s):
    return unescape_(s).replace('&quot;','"')

def getWordAndLineBoundaries(xmlPath):
    lines=[]
    w_lines=[]
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    writer = root.attrib['writer-id']
    allHs=0
    for line in root.findall('./handwritten-part/line'):

        trans=unescape(line.attrib['text'])
        minX=99999999
        maxX=-1
        minY=99999999
        maxY=-1
        words=[]
        for word in line.findall('word'):
            w_trans=unescape(word.attrib['text'])
            w_id=word.attrib['id']
            #print(w_trans)
            w_minX=99999999
            w_maxX=-1
            w_minY=99999999
            w_maxY=-1
            for cmp in word.findall('cmp'):
                x = int(cmp.attrib['x'])
                y = int(cmp.attrib['y'])
                w = int(cmp.attrib['width'])
                h = int(cmp.attrib['height'])
                #option1
                #maxX = max(maxX,x+w//2)
                #minX = min(minX,x-w//2)
                #maxY = max(maxY,y+h//2)
                #minY = min(minY,y-h//2)
                #option2
                maxX = max(maxX,x+w)
                minX = min(minX,x)
                maxY = max(maxY,y+h)
                minY = min(minY,y)
                w_maxX = max(w_maxX,x+w)
                w_minX = min(w_minX,x)
                w_maxY = max(w_maxY,y+h)
                w_minY = min(w_minY,y)
            words.append(([w_minY,w_maxY+1,w_minX,w_maxX+1],w_trans,w_id))

        
        #lineImg = formImg[minY:maxY+1,minX:maxX+1]
        lines.append(([minY,maxY+1,minX,maxX+1],trans))
        w_lines.append(words)
        allHs+=1+maxY-minY
    meanH = allHs/len(lines)
    newLines=[]
    for bounds,trans in lines:
        diff = meanH-(bounds[1]-bounds[0])
        if diff>0:
            bounds[0]-=diff/2
            bounds[1]+=diff/2
        bounds[2]-= meanH/4
        bounds[3]+= meanH/4
        bounds = [round(v) for v in bounds]
        #lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        newLines.append((bounds,trans))
    newW_lines=[]
    for words in w_lines:
        newWords=[]
        for bounds,trans,id in words:
            diff = meanH-(bounds[1]-bounds[0])
            if diff>0:
                bounds[0]-=diff/2
                bounds[1]+=diff/2
            bounds[2]-= meanH/4
            bounds[3]+= meanH/4
            bounds = [round(v) for v in bounds]
            #lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            newWords.append((bounds,trans,id))
        newW_lines.append(newWords)
    return  newW_lines,newLines, writer

def getLineBoundaries(xmlPath):
    lines=[]
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    writer = root.attrib['writer-id']
    allHs=0
    for line in root.findall('./handwritten-part/line'):

        trans=unescape(line.attrib['text'])
        assert('&' not in trans or ';' not in trans)
        minX=99999999
        maxX=-1
        minY=99999999
        maxY=-1
        for word in line.findall('word'):
            for cmp in word.findall('cmp'):
                x = int(cmp.attrib['x'])
                y = int(cmp.attrib['y'])
                w = int(cmp.attrib['width'])
                h = int(cmp.attrib['height'])
                #option1
                #maxX = max(maxX,x+w//2)
                #minX = min(minX,x-w//2)
                #maxY = max(maxY,y+h//2)
                #minY = min(minY,y-h//2)
                #option2
                maxX = max(maxX,x+w)
                minX = min(minX,x)
                maxY = max(maxY,y+h)
                minY = min(minY,y)

        
        #lineImg = formImg[minY:maxY+1,minX:maxX+1]
        lines.append(([minY,maxY+1,minX,maxX+1],trans))
        allHs+=1+maxY-minY
    meanH = allHs/len(lines)
    newLines=[]
    for bounds,trans in lines:
        diff = meanH-(bounds[1]-bounds[0])
        if diff>0:
            bounds[0]-=diff/2
            bounds[1]+=diff/2
        bounds[2]-= meanH/4
        bounds[3]+= meanH/4
        bounds = [round(v) for v in bounds]
        #lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        newLines.append((bounds,trans))
    return newLines, writer

def getLineBoundariesWithID(xmlPath):
    lines=[]
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    writer = root.attrib['writer-id']
    allHs=0
    for line in root.findall('./handwritten-part/line'):
        line_id = line.attrib['id']
        trans=unescape(line.attrib['text'])
        assert('&' not in trans or ';' not in trans)
        minX=99999999
        maxX=-1
        minY=99999999
        maxY=-1
        for word in line.findall('word'):
            for cmp in word.findall('cmp'):
                x = int(cmp.attrib['x'])
                y = int(cmp.attrib['y'])
                w = int(cmp.attrib['width'])
                h = int(cmp.attrib['height'])
                #option1
                #maxX = max(maxX,x+w//2)
                #minX = min(minX,x-w//2)
                #maxY = max(maxY,y+h//2)
                #minY = min(minY,y-h//2)
                #option2
                maxX = max(maxX,x+w)
                minX = min(minX,x)
                maxY = max(maxY,y+h)
                minY = min(minY,y)

        
        #lineImg = formImg[minY:maxY+1,minX:maxX+1]
        lines.append(([minY,maxY+1,minX,maxX+1],trans,line_id))
        allHs+=1+maxY-minY
    meanH = allHs/len(lines)
    newLines=[]
    for bounds,trans,line_id in lines:
        #if id==line_id:
        diff = meanH-(bounds[1]-bounds[0])
        if diff>0:
            bounds[0]-=diff/2
            bounds[1]+=diff/2
        bounds[2]-= meanH/4
        bounds[3]+= meanH/4
        bounds = [round(v) for v in bounds]
        #lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        newLines.append((bounds,trans,line_id))
    return newLines, writer
    #        return bounds,trans,writer
    #raise ValueError('id {} not in {}'.format(id,xmlPath))

def getLines(imagePath,xmlPath):
    formImg = imageio.imread(imagePath)
    lines=[]
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    writer = root.attrib['writer-id']
    print(writer)
    allHs=0
    for line in root.findall('./handwritten-part/line'):

        trans=unescape(line.attrib['text'])
        minX=99999999
        maxX=-1
        minY=99999999
        maxY=-1
        for word in line.findall('word'):
            for cmp in word.findall('cmp'):
                x = int(cmp.attrib['x'])
                y = int(cmp.attrib['y'])
                w = int(cmp.attrib['width'])
                h = int(cmp.attrib['height'])
                #option1
                #maxX = max(maxX,x+w//2)
                #minX = min(minX,x-w//2)
                #maxY = max(maxY,y+h//2)
                #minY = min(minY,y-h//2)
                #option2
                maxX = max(maxX,x+w)
                minX = min(minX,x)
                maxY = max(maxY,y+h)
                minY = min(minY,y)

        
        #lineImg = formImg[minY:maxY+1,minX:maxX+1]
        lines.append(([minY,maxY+1,minX,maxX+1],trans))
        allHs+=1+maxY-minY
    meanH = allHs/len(lines)
    newLines=[]
    for bounds,trans in lines:
        diff = meanH-(bounds[1]-bounds[0])
        if diff>0:
            bounds[0]-=diff/2
            bounds[1]+=diff/2
        bounds[2]-= meanH/4
        bounds[3]+= meanH/4
        bounds = [round(v) for v in bounds]
        lineImg = formImg[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        newLines.append((lineImg,trans))
    return newLines

def getWordAndLineIDs(xmlPath):
    lines=[]
    words=[]
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    writer = root.attrib['writer-id']
    allHs=0
    for line in root.findall('./handwritten-part/line'):

        line_id=line.attrib['id']
        lines.append(line_id)
        for word in line.findall('word'):
            w_id=word.attrib['id']
            w_trans=unescape(word.attrib['text'])
            words.append((w_id,w_trans,line_id))
    return words,lines

if __name__ == "__main__":
    #imagePath = sys.argv[1]
    #xmlPath = sys.argv[2]
    #lines = getLines(imagePath,xmlPath)
    #for i,(img, trans) in enumerate(lines):
    #    print('{}: {}'.format(i,trans))
    #    imageio.imsave('test/{}.png'.format(i),img)
    #lineOut = sys.argv[1]
    wordOut = sys.argv[1]
    idIn = sys.argv[2]
    vocabIn = sys.argv[3]

    valid_ids=[]
    with open(idIn) as f:
        for line in f.readlines():
            valid_ids.append(line[line.find('-')+1:line.rfind('-')])
    valid_vocab=[]
    with open(vocabIn) as f:
        for line in f.readlines():
            w,i = line.split(',')
            valid_vocab.append(w)
    #import pdb;pdb.set_trace()
    #words=defaultdict(list)
    words=[]
    lines=[]
    for xmlPath in sys.argv[4:]:
        ws,ls = getWordAndLineIDs(xmlPath)
        #words += ws
        lines += ls
        for id,trans,line_id in ws:
            #if trans=='.':
            #    import pdb;pdb.set_trace()
            #words[trans].append(id)
            if trans in valid_vocab:# and line_id in valid_ids:
                words.append(id)
                if id=='p06-069-00-04':
                    import pdb;pdb.set_trace()
                    print('WHAT!?')
            #else:
            #    print('rejected {} {}'.format(id,trans))
    #final_words = []
    #for trans,ids in words.items():
        #if len(ids)>20:

        #    final_words+=ids
    #words = final_words


    from random import shuffle
    shuffle(words)
    #shuffle(lines)

    #linesForTrain = int((4/6)*len(lines))
    #linesForValid = int((5/6)*len(lines))
    #linesTrain = lines[:linesForTrain]
    #linesValid = lines[linesForTrain:linesForValid]
    #linesTest = lines[linesForValid:]
    #print('num lines train: {},  valid: {},  test:  {}'.format(len(linesTrain),len(linesValid),len(linesTest)))

    wordsForTrain = int(0.62*len(words))
    wordsForValid = int(0.7138*len(words))
    wordsTrain = words[:wordsForTrain]
    wordsValid = words[wordsForTrain:wordsForValid]
    wordsTest = words[wordsForValid:]
    print('num words train: {},  valid: {},  test:  {}'.format(len(wordsTrain),len(wordsValid),len(wordsTest))) 

    #with open(lineOut,'w') as f:
    #    json.dump({'train':linesTrain,'valid':linesValid,'test':linesTest},f)
    with open(wordOut,'w') as f:
        json.dump({'train':wordsTrain,'valid':wordsValid,'test':wordsTest},f)
