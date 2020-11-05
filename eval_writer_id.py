import sys
import numpy as np
from glob import glob
import pickle


#def getEmbeddings(dataset,trainer):
#    ret={}
#    for instance in dataset:
#        pred, recon, losses, style = trainer.run(instance,get_style=True)
#        for i,author in enumerate(instance['author']):
#            ret[author] = style[i]
#    return ret

def topN(n,scores,gt):
    total=0
    for i in range(scores.shape[0]):
        paired = list(zip(scores[i],gt[i]))
        paired.sort(key=lambda a: a[0])
      
        #paired.reverse()
        #paired = np.stack((scores[i],gt[i]),axis=1)
        #paired = np.sort
        t=0
        for i in range(1,n+1):
            if paired[i][1]:
                t+=1
        total += t>0
    return total/scores.shape[0]
def bestTrue(scores,gt):
    total=0
    for i in range(scores.shape[0]):
        paired = list(zip(scores[i],gt[i]))
        #paired.sort(lambda a,b: a[0]<b[0])
        paired.sort(key=lambda a: a[0])
        #paired = np.stack((scores[i],gt[i]),axis=1)
        #paired = np.sort
        for rank in range(1,len(paired)):
            if paired[i][1]:
                break
        total += rank
    return total/scores.shape[0]

style_loc = sys.argv[1]
if style_loc[-1]!='*':
    style_loc+='*'
styles=[]
authors=[]
for loc in glob(style_loc):
    with open(loc,'rb') as f:
        data = pickle.load(f)
    s=data['styles']
    if len(s.shape)==4:
        s=s[:,:,0,0] 
    styles.append(s)
    
    authors+=list(data['authors'])
styles = np.concatenate(styles,axis=0)

print('styles: {}'.format(styles.shape))
#compute distance

#styles_expand = np.repeat(styles[None,:,:],styles.shape[0],axis=0)
#diff = styles_expand - np.transpose(styles_expand,(1,0,2))
#print('diff: {}'.format(diff))
#
#l2 = np.power(diff,2).sum(axis=2)
#l1 = np.abs(diff).sum(axis=2)

l1 = np.empty((styles.shape[0],styles.shape[0]))
l2 = np.empty((styles.shape[0],styles.shape[0]))
for i in range(styles.shape[0]):
    for j in range(styles.shape[0]):
        diff = styles[i]-styles[j]
        l1[i,j] = np.abs(diff).sum()
        l2[i,j] = np.power(diff,2).sum()

gt = np.empty((styles.shape[0],styles.shape[0]),dtype=np.int8)
for i,author1 in enumerate(authors):
    for j,author2 in enumerate(authors):
        gt[i,j] = author1==author2

#uptrI = np.triu_indices(styles.shape[0])
#l2 = l2[uptrI]
#l1 = l1[uptrI]
#gt = gt[uptrI]

print('l2 rank: {}'.format(bestTrue(l2,gt)))
print('l2\ttop1:{},\ttop5:\t{},\ttop20:\t{}'.format(topN(1,l2,gt),topN(5,l2,gt),topN(20,l2,gt)))
print('l1 rank: {}'.format(bestTrue(l1,gt)))
print('l1\ttop1:{},\ttop5:\t{},\ttop20:\t{}'.format(topN(1,l1,gt),topN(5,l1,gt),topN(20,l1,gt)))
