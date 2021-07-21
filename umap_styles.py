import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
import umap
import pickle, sys, random
from collections import defaultdict
from glob import iglob
import os

#sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
np.random.seed(42)
random.seed(42)

#mean = int(sys.argv[2]) if len(sys.argv)>2 else 0
mean=0
legend=False

images_path = sys.argv[2] if len(sys.argv)>2 else None
if images_path is not None:
    with open(os.path.join(images_path,'ordered.txt')) as f:
        images = f.readlines()
        per_author = int(images[0])
        images = images[1:]
        images = [os.path.join(images_path,s.strip()) for s in images]
else:
    images = None


addGaus = sys.argv[3]=='add' if len(sys.argv)>3 else False

style_loc = sys.argv[1]
if style_loc[-1]!='*':
    style_loc+='*'
styles=[]
authors=[]
for loc in iglob(style_loc):
    with open(loc,'rb') as f:
        data = pickle.load(f)
    s=data['styles']
    if type(s) is list:
        new_s=[]
        for style in s:
            style = np.concatenate([style[0],style[1],style[2].flatten()])
            new_s.append(style)
        s = np.stack(new_s,axis=0)
    if len(s.shape)==4:
        s=s[:,:,0,0]
    styles.append(s)

    authors+=list(data['authors'])
if addGaus:
    styles.append( np.random.normal(size=(100,styles[0].shape[1])) )
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




#color_map={}
#for a in authors:
#    if a not in color_map:
#        color_map[a] = np.random.rand(3)
color_map = defaultdict(lambda: np.random.rand(3))
markers = 'ov^<>12348spP*hH+xXdD'
marker_map = defaultdict(lambda: random.choice(markers))

#colors = [color_map[a] for a in authors]
#if addGaus:
#    colors += [np.array([0.0,0.0,0.0])]*100
print(styles.shape)

def drawMap(n_neighbors, min_dist):
    fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
    u = fit.fit_transform(styles)

    author_xs=defaultdict(list)
    author_ys=defaultdict(list)
    xys=[]

    author_count = defaultdict(lambda:0)

    for i in range(u.shape[0]):
        author_xs[authors[i]].append(u[i,0])
        author_ys[authors[i]].append(u[i,1])
        if images is not None and author_count[authors[i]]<per_author:
            xys.append(u[i])
            author_count[authors[i]]+=1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')   
    for author in author_xs:
        ax.scatter(author_xs[author],author_ys[author], c=[color_map[author]], marker=marker_map[author], label=author)

    if images is not None:
        artists = []
        for i,(x,y) in enumerate(xys): 
            image = plt.imread(images[i])
            im = OffsetImage(image, zoom=0.1)
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        #ax.update_datalim(np.column_stack([xs, ys]))

    #for i in range(u.shape[0]):
    #    ax.scatter(u[i,0], u[i,1], c=[colors[i]], label=authors[i])
    #ax.scatter(u[:,0], u[:,1], c=colors, label=authors)
    if legend:
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.7, chartBox.height])
        ax.legend(ncol=4,loc='upper right', bbox_to_anchor=(1.4, 1.0))
    #ax.title('Styles by author/font. nn={} min_dist={}'.format(n_neighbors,min_dist))


#nns = list(range(5,65,30))
#dists = np.arange(0,0.6,0.2)
nns=[35]
dists=[0.2]

for nn in nns:
    for dis in dists:
        drawMap(nn,dis)

plt.show()
