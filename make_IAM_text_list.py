import os,sys,json
from utils.parseIAM import getLineBoundaries as parseXML


dirPath = sys.argv[1]
outPath = sys.argv[2]

with open(os.path.join('data','sets.json')) as f:
    set_list = json.load(f)['test']

texts=[]
for page_idx, name in enumerate(set_list):
    lines,author = parseXML(os.path.join(dirPath,'xmls',name+'.xml'))
    texts += [t for b,t in lines]

with open(outPath,'w') as out:
    for t in texts:
        out.write('{}\n'.format(t))
