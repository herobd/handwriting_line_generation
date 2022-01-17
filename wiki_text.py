from datasets import load_dataset, load_from_disk
import random
import os
from utils.util import ensure_dir
import re
import unicodedata

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
        
        self.words=[]
        self.index=0

        self.genchars=set([" ", "!", "\"", "#", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"  , "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b",   "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])

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

        ret_c=0
        ret=[]
        while ret_c<50:
            if len(self.words)==0:
                self.addWords()
                if len(ret)>0:
                    break
            ret.append(self.words.pop(0))
            ret_c+=len(ret[-1])+1
        
        #ret = self.words[0]
        #if len(ret)>50:
        #    words = ret.split(' ')
        #    first=[]
        #    new_chars=0
        #    index=0
        #    for w in words:
        #        first.append(w)
        #        new_chars += len(w)+1
        #        index+=1
        #        if new_chars>50:
        #            break
        #    ret = ' '.join(first)
        #    self.words[0]=' '.join(
        #    ret = ret[:50]
        #else:
        #    self.words = self.sentences[1:]
        return ' '.join(ret)

    def addWords(self):
        while len(self.words)==0:
            article = self.getWikiArticle(all_newline=True)
            for para in article:
                #sents = [s.strip() for s in re.split('[.?!]',para)]
                #self.words+=[s+'.' for s in sents if len(s)>0 and s!=' ']
                words = [self.wordProcess(w) for w in re.split('[ \n]',para)]
                self.words+=[w for w in words if len(w)>0]

    def wordProcess(self,word):
        p = remove_accents(word)
        return ''.join(c for c in p if (c in self.genchars))
        #return p

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode("utf-8")
