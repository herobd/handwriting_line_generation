from datasets import load_dataset, load_from_disk
import random
import os
from utils.util import ensure_dir
import re
#import unicodedata

class Wikipedia:
    def __init__(self):

        if os.path.exists('DIR'):
            with open('DIR') as f:
                cache_path = f.readline().strip()
        else:
            cache_path = '../data/wiki_cache' #/Data6/davis/data_cache
            ensure_dir(cache_path)

        #if 'myVar' in locals():
        #_text_data = []#load_dataset('wikipedia', '20200501.en', cache_dir=cache_path)['train']

        self._prune_headers = ["See also", "Gallery", "External media", "History", "Notes"]
        self._wiki_end_keywords = ['References','Sources','External links']
        self._wiki_end_keywords = ['\n'+k+'\n' for k in self._wiki_end_keywords] + ['\n'+k+' \n' for k in self._wiki_end_keywords] + ['\nCategory:']

        #Returns a list of text paragraphs from a randome wikipedia article
        if not os.path.exists(os.path.join(cache_path,'dataset_info.json')):
            self._text_data = load_dataset('wikipedia', '20200501.en', cache_dir=cache_path)['train']
            self._text_data.save_to_disk(cache_path)
        else:
            self._text_data = load_from_disk(cache_path)
        
        self.sentences=[]
        self.index=0

    def getWikiArticle(self,all_newline=False):
        #Returns a list of text paragraphs from a randome wikipedia article


        #instance_i = random.randrange(self._text_data.num_rows)
        instance_i = self.index
        #print('Wiki on index {}'.format(self.index))
        self.index+=1
        text = self._text_data[instance_i]['text']
        #text = unicodedata.normalize(text,'NFC')#.decode('utf')


        #We first want to cut off the end of the wikipedia article, which has the references and stuff 
        for keyword in self._wiki_end_keywords:
            cut_i = text.find(keyword)
            if cut_i>-1:
                break
        if cut_i>-1:
            text = text[:cut_i]

        #break by paragraph (double newline)
        text=re.sub(r' +',r' ',text)
        if all_newline:
            text=re.sub(r'\n+',r'\n',text)
            paras = text.split('\n')
        else:
            paras = text.split('\n\n')

        paras = [para for para in paras if para.strip() not in self._prune_headers]
        
        if len(paras)>0:
            return paras
        else:
            print('blank article:')
            print(text)
            print('------------')
            print(self._text_data[instance_i]['text'])
            return getWikiArticle(all_newline)


    def __getitem__(self,i):
        while len(self.sentences)==0:
            article = self.getWikiArticle(all_newline=True)
            for para in article:
                sents = [s.strip() for s in re.split('[.?!]',para)]
                self.sentences+=[s+'.' for s in sents if len(s)>0 and s!=' ']
        
        ret = self.sentences.pop()
        if len(ret)>50:
            self.sentences.append(ret[50:])
            ret = ret[:50]
        return ret
