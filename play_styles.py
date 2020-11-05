import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
import umap
import pickle, sys, math
from collections import defaultdict

#sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

with open(sys.argv[1],'rb') as f:
    data = pickle.load(f)

styles=data['styles']
if len(styles.shape)==4:
    styles=styles[:,:,0,0]
authors=data['authors']

inter_author_dist=[]
intra_author_dist=[]

#author_matrix=np.empty((len(authors),len(authors)))
author_d=defaultdict(list)

def dist(vA,vB):
    #return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
    #return np.abs(vA-vB).sum()
    return math.sqrt(np.power(vA-vB,2).sum())

for i in range(len(authors)):
    for j in range(i+1,len(authors)):
        d=dist(styles[i],styles[j])
        if authors[j]==authors[i]:
            inter_author_dist.append( d)
        else:
            intra_author_dist.append(d)
        author_d[(i,j)].append(d)
print('inter dist mean: {}, stddev: {}'.format(np.mean(inter_author_dist),np.std(inter_author_dist)))
print('intra dist mean: {}, stddev: {}'.format(np.mean(intra_author_dist),np.std(intra_author_dist)))

exit()

author_mean=np.zeros((len(authors),len(authors)))
author_std=np.zeros((len(authors),len(authors)))

for pair,l in author_d.items():
    a1,a2=pair
    m=np.mean(l)
    s=np.std(l)
    author_mean[a1,a2]=m
    author_mean[a2,a1]=m
    author_std[a1,a2]=s
    author_std[a2,a1]=s


cmap=plt.cm.Blues

fig, ax = plt.subplots()
im = ax.imshow(author_mean, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(author_mean.shape[1]),
           yticks=np.arange(author_mean.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title='mean',
           #ylabel='True label',
           #xlabel='Predicted label')
           )

fig, ax = plt.subplots()
im = ax.imshow(author_std, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(author_std.shape[1]),
           yticks=np.arange(author_std.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title='std',
           #ylabel='True label',
           #xlabel='Predicted label')
           )


plt.show()
