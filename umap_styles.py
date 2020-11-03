# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
import umap
import pickle, sys
from collections import defaultdict
from glob import iglob

#sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})


mean = int(sys.argv[2]) if len(sys.argv)>2 else 0

style_loc = sys.argv[1]
if style_loc[-1]!='*':
    style_loc+='*'
styles=[]
authors=[]
for loc in iglob(style_loc):
    with open(loc,'rb') as f:
        data = pickle.load(f)
    s=data['styles']
    if len(s.shape)==4:
        s=s[:,:,0,0]
    styles.append(s)

    authors+=list(data['authors'])
styles = np.concatenate(styles,axis=0)

if mean>0:
    by_author=defaultdict(list)
    for i in range(len(authors)):
        by_author[authors[i]].append(styles[i])
    #for author, ss in by_author.items():
    #    print('{} : {}'.format(author,len(ss)))
    #exit()
    new_styles=[]
    new_authors=[]
    for author, ss in by_author.items():
        i=0
        while len(ss)-i>=2*mean:
            summed = ss[i]
            i+=1
            count=0
            for j in range(1,mean):
                summed += ss[i]
                i+=1
                count+=1
            new_styles.append(summed/count)
            new_authors.append(author)
        summed = ss[i]
        i+=1
        count=0
        for j in range(i,len(ss)):
            summed += ss[j]
            count+=1
        new_styles.append(summed/count)
        new_authors.append(author)

        styles = np.stack(new_styles)
        authors = new_authors


np.random.seed(42)

#color_map={}
#for a in authors:
#    if a not in color_map:
#        color_map[a] = np.random.rand(3)
color_map = defaultdict(lambda: np.random.rand(3))

colors = [color_map[a] for a in authors]

def drawMap(n_neighbors, min_dist):
    fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
    u = fit.fit_transform(styles)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(u[:,0], u[:,1], c=colors)
    #ax.title('Styles by author/font. nn={} min_dist={}'.format(n_neighbors,min_dist))


nns = list(range(5,100,30))
dists = np.arange(0,0.6,0.2)

for nn in nns:
    for dis in dists:
        drawMap(nn,dis)

plt.show()
