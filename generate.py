import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from model.loss import *
from logger import Logger
#from trainer import *
#from data_loader import getDataLoader
from datasets.text_data import TextData
#from evaluators import *
import math
from collections import defaultdict
import pickle
from glob import glob
import cv2
from utils import string_utils
from utils.util import ensure_dir
import random, re, csv

#from datasets.forms_detect import FormsDetect
#from datasets import forms_detect

logging.basicConfig(level=logging.INFO, format='')

data_loader= None
valid_data_loader = None
authors = None

def permuteF(sent):
    s = sent.split(' ')
    if len(s)>4:
        m = s[1:len(s)-1]
        while m == s[1:len(s)-1]:
            random.shuffle(m)
        s = s[0:1]+m+s[len(s)-1:]
    elif len(s)>2:
        m = s[0:len(s)]
        while m == s:
            random.shuffle(m)
        s=m
    else:
        return 'Kevin Bacon'
    return ' '.join(s)

def get_style(config,model,instance, gpu=None):
    lookup_style = 'lookup' in config['model']['style'] or 'Lookup' in config['model']['style']
    style_together = config['trainer']['style_together'] if 'style_together' in config['trainer'] else False
    use_hwr_pred_for_style = config['trainer']['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config['trainer'] else False
    image = instance['image']
    label = instance['label']
    if gpu is not None:
        image = image.to(gpu)
        label = label.to(gpu)
    if lookup_style:
        style = model.style_extractor(instance['author'],gpu)
    else:
        if not style_together:
            style = model.style_extractor(image)
            style = style[0:1]
        else:
            #append all the instances in the batch by the same author together along the width dimension
            pred = model.hwr(image, None)
            num_class = pred.size(2)
            if use_hwr_pred_for_style:
                spaced_label = pred.permute(1,2,0)
            else:
                spaced_label = correct_pred(pred,label)
                spaced_label = onehot(spaced_label).permute(1,2,0)
            batch_size,feats,h,w = image.size()
            if 'a_batch_size' in instance:
                a_batch_size = instance['a_batch_size']
            else:
                a_batch_size = batch_size
            spaced_len = spaced_label.size(2)
            collapsed_image =  image.permute(1,2,0,3).contiguous().view(feats,h,batch_size//a_batch_size,w*a_batch_size).permute(2,0,1,3)
            collapsed_label = spaced_label.permute(1,0,2).contiguous().view(num_class,batch_size//a_batch_size,spaced_len*a_batch_size).permute(1,0,2)
            style = model.style_extractor(collapsed_image, collapsed_label)
            #style=style.expand(batch_size,-1)
            #style = style.repeat(a_batch_size,1)
    return style

def main(resume,saveDir,gpu=None,config=None,addToConfig=None, fromDataset=True, test=False, arguments=None,style_loc=None):
    np.random.seed(1234)
    torch.manual_seed(1234)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded iteration {}'.format(checkpoint['iteration']))
        ##HACK fix
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            if 'style_from_normal' in key: #HACK
                del checkpoint['state_dict'][key]
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        for key in config.keys():
            if 'pretrained' in key:
                config[key]=None
    else:
        checkpoint = None
        config = json.load(open(config))
    config['model']['RUN']=True
    config['optimizer_type']="none"
    config['trainer']['use_learning_schedule']=False
    config['trainer']['swa']=False
    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    addDATASET=False
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            printM='added config['
            for i in range(len(add)-2):
                addTo = addTo[add[i]]
                printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            addTo[add[-2]] = value
            printM+=add[-2]+']={}'.format(value)
            print(printM)
            if (add[-2]=='useDetections' or add[-2]=='useDetect') and value!='gt':
                addDATASET=True

    if fromDataset:
        config['data_loader']['batch_size']=1
        config['validation']['batch_size']=1
        def get_valid_data_loader():
            global data_loader
            global valid_data_loader
            if valid_data_loader is None:
                if not test:
                    data_loader, valid_data_loader = getDataLoader(config,'train')
                else:
                    config['data_loader']['a_batch_size']=1
                    config['validation']['a_batch_size']=1
                    print('changed a_batch_size to 1')
                    test_data_loader, _ = getDataLoader(config,'test')
                    valid_data_loader = test_data_loader

            return valid_data_loader
        def get_data_loader():
            global data_loader
            global valid_data_loader
            if valid_data_loader is None:
                if not test:
                    data_loader, valid_data_loader = getDataLoader(config,'train')
                else:
                    config['data_loader']['a_batch_size']=1
                    config['validation']['a_batch_size']=1
                    print('changed a_batch_size to 1')
                    test_data_loader, _ = getDataLoader(config,'test')
                    valid_data_loader = test_data_loader

            return data_loader
        def get_test_data_loader():
            global data_loader
            global valid_data_loader
            if valid_data_loader is None:
                if not test:
                    data_loader, valid_data_loader = getDataLoader(config,'train')
                else:
                    config['data_loader']['a_batch_size']=1
                    config['validation']['a_batch_size']=1
                    print('changed a_batch_size to 1')
                    test_data_loader, _ = getDataLoader(config,'test')
                    valid_data_loader = test_data_loader

            return valid_test_loader
    

    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = eval(config['arch'])(config['model'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    else:
        model = eval(config['arch'])(config['model'])
    model.eval()
    #model.summary()
    if gpu is not None:
        model = model.to(gpu)
    model.count_std=0
    model.dup_std=0

    gt_mask = 'create_mask' not in config['model'] #'mask' in config['model']['generator'] or 'Mask' in config['model']['generator']

    char_set_path = config['data_loader']['char_file']
    if char_set_path=='../data/RIMES/characterset_lines.json':
        char_set_path='data/RIMES_characterset_lines.json'
    with open(char_set_path) as f:
        char_set = json.load(f)
    char_to_idx = char_set['char_to_idx']


    by_author_styles=defaultdict(list)
    by_author_all_ids=defaultdict(set)
    #style_loc = config['style_loc'] if 'style_loc' in config else style_loc
    if style_loc is not None:
        if style_loc[-1]!='*':
            style_loc+='*'
        all_style_files = glob(style_loc)
        assert( len(all_style_files)>0)
        for loc in all_style_files:
            #print('loading '+loc)
            with open(loc,'rb') as f:
                styles = pickle.load(f)
            if 'ids' in styles:
                for i in range(len(styles['authors'])):
                    by_author_styles[styles['authors'][i]].append((styles['styles'][i],styles['ids'][i]))
                    by_author_all_ids[styles['authors'][i]].update(styles['ids'][i])
            else:
                for i in range(len(styles['authors'])):
                    by_author_styles[styles['authors'][i]].append((styles['styles'][i],None))

        styles = defaultdict(list)
        authors=set()
        for author in by_author_styles:
            for style, ids in by_author_styles[author]:
                    styles[author].append(style)
            if len(styles[author])>0:
                authors.add(author)
        authors=list(authors)
    elif not test:
        authors = None
        styles = None
    else:
        styles = None

    def get_authors():
        global authors
        if authors is None:
            authors = get_valid_data_loader().dataset.authors
        return authors

    num_char = config['model']['num_class']
    use_hwr_pred_for_style = config['trainer']['use_hwr_pred_for_style'] if 'use_hwr_pred_for_style' in config['trainer'] else False

    charSpec = model.char_style_dim>0

    with torch.no_grad():
        while True:
            if arguments is None:
                action = input('indexes/random interp/vae random/strech/author display/math/turk gen/from-to/umap-images/Random styles/help/quit? ') #indexes/random/vae/strech/author-list/quit
            else:
                action = arguments['choice']
                arguments['choice']='q'
            if action=='done' or action=='exit' or 'action'=='quit' or action=='q':
                exit()
            elif action[0]=='h': #help
                print('Options:')
                print('[a] show author ids')
                print('[r] random interpolation: selects n styles (dataset extracted) and interpolated between them in a circlular pattern')
                print('[v] same as above, but styles are randomly sampled from guassian distribution (for VAE)')
                print('[s] strech: manipulate the 1d text encoding to interpolate horizontal streching')
                print('[m] vector math: perform vector math with style vectors. Use "+" and "-". Use author id to specifiy vector, and [id1,id2,...] to average vectors together.')
                print('[t] MTurk gen: routine used to generate data for MTurk experimenet')
                print('[R] Random: generate n images using random (interpolated) styles. Can use fixed or random text')
                print("[f] Given two image paths, interpolate from one style to the other using the given text.")
            elif action =='a' or action=='authors':
                print(get_authors())
            elif action =='s' or action=='strech':
                index1=input("batch? ")
                if len(index1)>0:
                    index1=int(index1)
                else:
                    index1=0
                for i,instance1 in enumerate(get_valid_data_loader()):
                    if i==index1:
                        break
                author1 = instance1['author'][0]
                style1 = get_style(config,model,instance1,gpu)
                image = instance1['image']
                label = instance1['label']
                if gpu is not None:
                    image = image.to(gpu)
                    label = label.to(gpu)
                pred=model.hwr(image, None)
                if use_hwr_pred_for_style:
                    spaced_label = pred
                else:
                    spaced_label = model.correct_pred(pred,label)
                    spaced_label = model.onehot(spaced_label)
                images=interpolate_horz(model,style1, spaced_label)
                for b in range(images[0].size(0)):
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        cv2.imwrite(path,genStep)
            elif action[0]=='r' or action[0]=='v': #interpolate randomly selected styles, "v" is VAE
                num_styles = int(input('number of styles? '))
                step = float(input('step (0.1 is normal)? '))
                text = input('text? ')
                if len(text)==0:
                    text='The quick brown fox jumps over the lazy dog.'
                stylesL=[]
                if action[0]=='r':
                    index = random.randint(0,20)
                    last_author = None
                    for i,instance in enumerate(get_valid_data_loader()):
                        author = instance['author'][0]
                        if i>=index and author!=last_author:
                            print('i: {}, a: {}'.format(i,author))
                            image=instance['image'].to(gpu)
                            label=instance['label'].to(gpu)
                            a_batch_size = instance['a_batch_size']
                            style=model.extract_style(image,label,a_batch_size)[::a_batch_size]
                            stylesL.append(style)
                            last_author=author
                            index += random.randint(20,50)
                            print('next index: {}'.format(index))
                        if len(stylesL)>=num_styles:
                            break
                else: #VAE
                    stylesL=[torch.FloatTensor(1,model.style_dim).normal_() for i in range(num_styles)]
                images=[]
                styles=[]
                #step=0.05
                for i in range(num_styles-1):
                    b_im, b_sty = interpolate(model,stylesL[i].to(gpu),stylesL[i+1].to(gpu), text,char_to_idx,gpu,step)
                    images+=b_im
                    styles+=b_sty
                b_im, b_sty = interpolate(model,stylesL[-1].to(gpu),stylesL[0].to(gpu), text,char_to_idx,gpu,step)
                images+=b_im
                styles+=b_sty
                for b in range(images[0].size(0)):
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        if step==0.2 and i%5==0:
                            genStep[0,:]=0
                            genStep[-1,:]=0
                            genStep[:,0]=0
                            genStep[:,-1]=0
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        #print('wrote: {}'.format(path))
                        cv2.imwrite(path,genStep)
                    torch.save(styles, os.path.join(saveDir,'styles{}.pth'.format(b)))
            
            elif action[0]=='R': #Just random (interpolated) styles, with option for random text
                assert(styles is not None and 'perhaps you forgot to set "-s path/to/styles.pkl"?')
                num_inst = int(input('num to gen? '))
                text = input('text? (enter "RANDOM" or "WIKI" or file path (.txt) for sampled text"): ') 
                index_offset=0
                if len(text)==0:
                    text='The quick brown fox jumps over the lazy dog.'
                    textList = None
                elif text=='RANDOM':
                    text=None
                    textData = TextData(batch_size=num_inst,max_len=55)
                    textList = textData.getInstance()['gt']
                elif text=='WIKI':
                    from wiki_text import Wikipedia
                    textList = Wikipedia()
                    index_offset = int(input('index start:'))
                    for i in range(index_offset):
                        text = textList[i]
                elif text.endswith('.txt'):
                    textData = TextData(batch_size=num_inst,max_len=55,textfile=text)
                    textList = textData.getInstance()['gt']
                    text=None
                else:
                    textList=None

                #sample the styles
                stylesL=[]
                textL=[]
                text_falseL=[]
                ensure_dir(os.path.join(saveDir))#,'fake'))
                for i in range(num_inst):
                    if not model.vae:
                        #authorA = random.choice(get_authors())
                        authorA = random.choice(list(styles.keys()))
                        instance = random.randint(0,len(styles[authorA])-1)
                        style1 = styles[authorA][instance]
                        #authorB = random.choice(get_authors())
                        authorB = random.choice(list(styles.keys()))
                        instance = random.randint(0,len(styles[authorB])-1)
                        style2 = styles[authorB][instance]

                        #inter = random.random()
                        inter = 2*random.random()-0.5
                        if charSpec:
                            style = (
                                    style1[0]*inter + style2[0]*(1-inter),
                                    style1[1]*inter + style2[1]*(1-inter),
                                    style1[2]*inter + style2[2]*(1-inter)
                                    )
                        else:
                            style = style1*inter + style2*(1-inter)

                        #stylesL.append(style)
                    else: #VAE
                        stylesL=[torch.FloatTensor(1,model.style_dim).normal_() for i in range(num_styles)]
                    #for i,style in enumerate(stylesL):
                    if charSpec:
                        if gpu is not None:
                            style = (torch.from_numpy(style[0])[None,...].to(gpu),torch.from_numpy(style[1][None,...]).to(gpu),torch.from_numpy(style[2][None,...]).to(gpu))
                        else:
                            style = (torch.from_numpy(style[0])[None,...],torch.from_numpy(style[1])[None,...],torch.from_numpy(style[2])[None,...])
                    else:
                        if gpu is not None:
                            style = torch.from_numpy(style).to(gpu)
                        else:
                            style = torch.from_numpy(style)
                        style = style[None,...]

                    if textList is not None:
                        text = textList[i]
                    im = generate(model,style,text,char_to_idx,gpu)
                    im = ((1-im[0].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                    image_name = 'sample_{}.png'.format(i+index_offset)
                    path = os.path.join(saveDir,image_name)
                    cv2.imwrite(path,im)
                    if textList is not None:
                        with open('OUT.txt','a') as out:
                            out.write('{}:'.format(i+index_offset)+text+'\n')


            elif action[0]=='m': #style vector math, this is broken
                assert(styles is not None and 'perhaps you forgot to set "-s path/to/styles.pkl"?')

                text = input('text? ')
                if len(text)==0:
                    text='The quick brown fox jumps over the lazy dog.'
                print('elements of expression: author_id,+,-,[author_id')
                expression = input('expression? ')
                idx=0
                #style=torch.FloatTensor(1,model.style_dim).zero_()
                m = re.search(r'^(\d+|\+|-|\[[^-\+]+\])',expression[idx:])
                segment=m[0]
                idx+=len(segment)
                if segment[0]=='[':
                    nums=[int(s) for s in segment[1:-1].split(',')]
                    style=styles[nums[0]]
                    for num in nums[1:]:
                        subStyle+=styles[num]
                    style /= len(nums)
                else:
                    #if normal:
                    style=styles[segment][0]
                    #else:
                    #    style=styles[int(segment)]
                while idx<len(expression):
                    m = re.search(r'^(\d+|\+|-|\[[^-\+]+\])',expression[idx:])
                    operation=m[0]
                    idx+=len(operation)

                    m = re.search(r'^(\d+|\+|-|\[[^-\+]+\])',expression[idx:])
                    segment=m[0]
                    idx+=len(segment)
                    if segment[0]=='[':
                        nums=[int(s) for s in segment[1:-1].split(',')]
                        subStyle=styles[nums[0]]
                        for num in nums[1:]:
                            subStyle+=styles[num]
                        subStyle /= len(nums)
                    else:
                        subStyle=styles[int(segment)]
                    if operation=='+':
                        style+=subStyle
                    elif operation=='-':
                        style+=subStyleS

                #if normal:
                if type(style) is list:
                    if gpu is not None:
                        style = (torch.from_numpy(style[0])[None,...].to(gpu),torch.from_numpy(style[1][None,...]).to(gpu),torch.from_numpy(style[2][None,...]).to(gpu))
                    else:
                        style = (torch.from_numpy(style[0])[None,...],torch.from_numpy(style[1])[None,...],torch.from_numpy(style[2])[None,...])
                else:
                    if gpu is not None:
                        style = torch.from_numpy(style).to(gpu)
                    else:
                        style = torch.from_numpy(style)
                    style = style[None,...]
                #else:
                #    style=style.to(gpu)

                im=generate(model,style.to(gpu), text,char_to_idx, gpu)
                im = ((1-im[0].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                path = os.path.join(saveDir,'result.png')
                cv2.imwrite(path,im)


            elif action=='A': #average an authors style vectors together
                author=input("author? ")
                text=input("text? ")
                if len(text)==0:
                    text='The quick brown fox jumps over the lazy dog.'
                max_hits =input("max instances? ")
                if len(max_hits)>0:
                    max_hits=int(max_hits)
                else:
                    max_hits=5
                styles=[]
                for i,instance1 in enumerate(get_valid_data_loader):
                    if instance1['author'][0]==author:
                        print('{} found on instance {}'.format(author,i))
                        label1 = instance1['label'].to(gpu)
                        image1 = instance1['image'].to(gpu)
                        a_batch_size = instance1['a_batch_size']
                        styles.append( model.extract_style(image1,label1,a_batch_size)[::a_batch_size])
                        max_hits-=1
                        if max_hits<=0:
                            break
                styles = torch.cat(styles,dim=0)
                style = styles.mean(dim=0)[None,...]
                im = generate(model,style,text,char_to_idx,gpu)
                im = ((1-im[0].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                path = os.path.join(saveDir,'gen_{}.png'.format(author))
                cv2.imwrite(path,im)

            elif action[0]=='t': #generate random samples for MTurk test.
                start_index=0
                assert(styles is not None and 'use -a style_loc')
                if arguments is None:
                    num_inst = input('number of instances? ')
                else:
                    num_inst = arguments['num_inst']
                    if 'start_index' in arguments: #this option is to start the image indexing later, so I could easily add the poorly generated images to the main set more easily
                        start_index=int(arguments['start_index'])
                num_inst = int(num_inst)

                if arguments is None:
                    interpolateS = input('interpolate? [Y]/N: ') #whether to interpolate styles, or take the directly from images
                elif 'interpolate' in arguments:
                    interpolateS = arguments['interpolate']
                else:
                    interpolateS = 'Y'
                interpolateS = interpolateS!='N' and interpolateS!='n'

                false_full=True

                stylesL=[]
                textL=[]
                text_falseL=[]
                #first build a list of styles
                for i in range(num_inst):
                    if not model.vae:
                        authorA = random.choice(get_authors())
                        instance = random.randint(0,len(styles[authorA])-1)
                        style1 = styles[authorA][instance]
                        if interpolateS:
                            authorB = random.choice(get_authors())
                            instance = random.randint(0,len(styles[authorB])-1)
                            style2 = styles[authorB][instance]

                            inter = random.random()
                            if charSpec:
                                style = (
                                        style1[0]*inter + style2[0]*(1-inter),
                                        style1[1]*inter + style2[1]*(1-inter),
                                        style1[2]*inter + style2[2]*(1-inter)
                                        )
                            else:
                                style = style1*inter + style2*(1-inter)
                        else:
                            style = style1

                        stylesL.append(style)
                    else: #VAE
                        stylesL=[torch.FloatTensor(1,model.style_dim).normal_() for i in range(num_styles)]
                images=[]
                ensure_dir(os.path.join(saveDir))
                to_write=[]
                with open(os.path.join(saveDir,'text.csv'),'w') as text_out:
                    #text.csv is the data for MTurk

                    #save the real images, from test set
                    for i in range(num_inst):
                        index = random.randint(0,len(get_test_data_loader())-1)
                        instance = test_data_loader.dataset[index]
                        text = instance['gt'][0]
                        textL.append(text)
                        while(True):
                            indexF = random.randint(0,len(test_data_loader)-1)
                            if indexF != index:
                                break
                        instanceF = test_data_loader.dataset[indexF]
                        textF = instanceF['gt'][0]
                        textF = permuteF(re.sub(r'[^\w\s]','',text))
                        im = ((1-instance['image'][0].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        image_name='sample_{}.png'.format(i+start_index)
                        path = os.path.join(saveDir,image_name)
                        cv2.imwrite(path,im)
                        url = 'http://students.cs.byu.edu/~brianld/images/{}'.format(image_name)

                        to_write.append([url,re.sub(r'[^\w\s]','',text),textF,image_name,'real'])
                    
                    random.shuffle(textL)
                    #save the fake generated images
                    for i,(style, text) in enumerate(zip(stylesL,textL)):
                        if charSpec:
                            if gpu is not None:
                                style = (torch.from_numpy(style[0])[None,...].to(gpu),torch.from_numpy(style[1][None,...]).to(gpu),torch.from_numpy(style[2][None,...]).to(gpu))
                            else:
                                style = (torch.from_numpy(style[0])[None,...],torch.from_numpy(style[1])[None,...],torch.from_numpy(style[2])[None,...])
                        else:
                            if gpu is not None:
                                style = torch.from_numpy(style).to(gpu)
                            else:
                                style = torch.from_numpy(style)
                            style = style[None,...]

                        im = generate(model,style,text,char_to_idx,gpu)
                        im = ((1-im[0].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        image_name = 'sample_{}.png'.format(i+num_inst+start_index)
                        path = os.path.join(saveDir,image_name)
                        cv2.imwrite(path,im)
                        url = 'http://students.cs.byu.edu/~brianld/images/{}'.format(image_name)
                        text = re.sub(r'[^\w\s]','',text)
                        textF = permuteF(text)
                        to_write.append([url,text,textF,image_name,'generated'])

                    random.shuffle(to_write)
                    csvwriter = csv.writer(text_out, delimiter=',',
                                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(['image_url','real_text','false_text','image_name','type'])
                    for l in to_write:
                        csvwriter.writerow(l)
                

                    
            elif action[0]=='f':  #from given image's style to other image's style.
                if arguments is None:
                    path1 = input("image path 1? ")
                    if len(path1)==0:
                        path1='real1.png'
                    path2 = input("image path 2? ")
                    if len(path2)==0:
                        path2='real2.png'
                    text_gen = input("text to generate? ")
                else:
                    path1 = arguments['path1']
                    path2 = arguments['path2']
                    text_gen = arguments['text_gen'] if 'text_gen' in arguments else arguments['text']
                img_height=64

                image1 = cv2.imread(path1,0)
                if image1.shape[0] != img_height:
                    percent = float(img_height) / image1.shape[0]
                    image1 = cv2.resize(image1, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
                image1 = image1[...,None]
                image1 = image1.astype(np.float32)
                image1 = 1.0 - image1 / 128.0
                image1 = image1.transpose([2,0,1])
                image1 = torch.from_numpy(image1)
                if gpu is not None:
                    image1=image1.to(gpu)

                image2 = cv2.imread(path2,0)
                if image2.shape[0] != img_height:
                    percent = float(img_height) / image2.shape[0]
                    image2 = cv2.resize(image2, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
                image2 = image2[...,None]
                image2 = image2.astype(np.float32)
                image2 = 1.0 - image2 / 128.0
                image2 = image2.transpose([2,0,1])
                image2 = torch.from_numpy(image2)
                if gpu is not None:
                    image2=image2.to(gpu)

                min_width = min(image1.size(2),image2.size(2))
                style = model.extract_style(torch.stack((image1[:,:,:min_width],image2[:,:,:min_width]),dim=0),None,1)
                if type(style) is tuple:
                    style1 = (style[0][0:1],style[1][0:1],style[2][0:1])
                    style2 = (style[0][1:2],style[1][1:2],style[2][1:2])
                else:
                    style1 = style[0:1]
                    style2 = style[1:2]

                images,stylesInter=interpolate(model,style1,style2, text_gen,char_to_idx,gpu)

                for b in range(images[0].size(0)):
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        #print('wrote: {}'.format(path))
                        cv2.imwrite(path,genStep)


            elif action[0]=='u': #Umap images, and image for every style, this was to replicate figure in GANWriting paper, but we didn't end up using it. May not work
                per_author=3
                text='deep'
                with open(os.path.join(saveDir,'ordered.txt'),'w') as f:
                    f.write('{}\n'.format(per_author))
                    for author in get_authors():
                        for i in range(per_author):
                            style = styles[author][i]
                            if charSpec:
                                if gpu is not None:
                                    style = (torch.from_numpy(style[0])[None,...].to(gpu),torch.from_numpy(style[1][None,...]).to(gpu),torch.from_numpy(style[2][None,...]).to(gpu))
                                else:
                                    style = (torch.from_numpy(style[0])[None,...],torch.from_numpy(style[1])[None,...],torch.from_numpy(style[2])[None,...])
                            else:
                                if gpu is not None:
                                    style = torch.from_numpy(style).to(gpu)
                                else:
                                    style = torch.from_numpy(style)
                                style = style[None,...]
                            im = generate(model,style,text,char_to_idx,gpu)
                            im = ((1-im[0].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                            image_name = '{}_{}.png'.format(author,i)
                            path = os.path.join(saveDir,image_name)
                            cv2.imwrite(path,im)
                            f.write('{}\n'.format(path))

            else:#if action=='i' or action=='interpolate':
                if fromDataset and styles is None:
                    index1=input("batch? ")
                    if len(index1)>0:
                        index1=int(index1)
                    else:
                        index1=0
                    if index1>=0:
                        data = get_valid_data_loader()
                    else:
                        index1*=-1
                        data = get_data_loader()
                    for i,instance1 in enumerate(data):
                        if i==index1:
                            break
                    author1 = instance1['author'][0]
                    print('author: {}'.format(author1))
                else:
                    author1=input("author? ")
                    if len(author1)==0:
                        author1=get_authors()[0]
                if True: #new way
                    mask=None
                    index=input("batch? ")
                    text=input("text? ")
                    if len(text)==0:
                        text='The quick brown fox jumps over the lazy dog.'
                    if len(index)>0:
                        index=int(index)
                    else:
                        index=0
                    if index>=0:
                        data = get_valid_data_loader()
                    else:
                        index*=-1
                        data = get_data_loader()
                    for i,instance2 in enumerate(data):
                        if i==index:
                            break
                    author2 = instance2['author'][0]
                    print('author: {}'.format(author2))
                    image1 = instance1['image'].to(gpu)
                    label1 = instance1['label'].to(gpu)
                    image2 = instance2['image'].to(gpu)
                    label2 = instance2['label'].to(gpu)
                    a_batch_size = instance1['a_batch_size']
                    #spaced_label = correct_pred(pred,label)
                    #spaced_label = onehot(spaced_label,num_char)
                    if styles is not None:
                        style1 = styles[author1][0]
                        style2 = styles[author2][0]
                        style1=torch.from_numpy(style1)
                        style2=torch.from_numpy(style2)
                    else:
                        style1 = model.extract_style(image1,label1,a_batch_size)[::a_batch_size]
                        style2 = model.extract_style(image2,label2,a_batch_size)[::a_batch_size]
                    images,stylesInter=interpolate(model,style1,style2, text,char_to_idx,gpu)

                if mask is not None:
                    mask = ((mask.cpu().permute(0,2,3,1)+1)/2.0).numpy()
                for b in range(images[0].size(0)):
                    for i in range(len(images)):
                        genStep = ((1-images[i][b].permute(1,2,0))*127.5).cpu().numpy().astype(np.uint8)
                        path = os.path.join(saveDir,'gen{}_{}.png'.format(b,i))
                        cv2.imwrite(path,genStep)

                    if mask is not None:
                        path_mask = os.path.join(saveDir,'mask{}.png'.format(b))
                        cv2.imwrite(path_mask,mask[b])


# generates an image
def generate(model,style,text,char_to_idx,gpu):
    #print(style)
    batch_size = 1#style.size(0)
    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:,None].expand(-1,batch_size).to(gpu).long()
    label_len = torch.IntTensor(batch_size).fill_(label.size(0))
    results=[]
    styles=[]
    return model(label,label_len,style)

# generates a series of images interpolating between the styles
def interpolate(model,style1,style2,text,char_to_idx,gpu,step=0.05):
    if type(style1) is tuple:
        batch_size = style1[0].size(0)
    else:
        batch_size = style1.size(0)
    label = string_utils.str2label_single(text, char_to_idx)
    label = torch.from_numpy(label.astype(np.int32))[:,None].expand(-1,batch_size).to(gpu).long()
    label_len = torch.IntTensor(batch_size).fill_(len(text))
    results=[]
    styles=[]
    for alpha in np.arange(0,1.0,step):
        if type(style1) is tuple:
            style = (style2[0]*alpha+(1-alpha)*style1[0],style2[1]*alpha+(1-alpha)*style1[1],style2[2]*alpha+(1-alpha)*style1[2])
        else:
            style = style2*alpha+(1-alpha)*style1
        gen = model(label,label_len,style)
        results.append(gen)
        if type(style) is tuple:
            styles.append((style[0].cpu().detach(),style[1].cpu().detach(),style[2].cpu().detach()))
        else:
            styles.append(style.cpu().detach())
    return results, styles

def interpolate_horz(model,style,spaced_label):
    results=[]
    style = style.view(1,-1).expand(spaced_label.size(1),-1)
    orig_spaced_label = spaced_label.permute(1,2,0)
    for strechH in np.arange(1,1.11,0.01):
        spaced_label = F.interpolate(orig_spaced_label,scale_factor=strechH,mode='linear').permute(2,0,1)
        gen = model.generator(spaced_label,style)
        results.append(gen)
    for strechV in np.arange(1,1.11,0.01):
        gen = model.generator(spaced_label,style)
        results.append(gen)
    for strechH in np.arange(1.1,0.89,-0.01):
        spaced_label = F.interpolate(orig_spaced_label,scale_factor=strechH,mode='linear').permute(2,0,1)
        gen = model.generator(spaced_label,style)
        results.append(gen)
    for strechV in np.arange(1.1,0.89,-0.01):
        gen = model.generator(spaced_label,style)
        results.append(gen)
    for strech in np.arange(0.9,1.01,0.01):
        spaced_label = F.interpolate(orig_spaced_label,scale_factor=strech,mode='linear').permute(2,0,1)
        gen = model.generator(spaced_label,style)
        results.append(gen)
    return results
def onehot(label,num_class):
    label_onehot = torch.zeros(label.size(0),label.size(1),num_class)
    #label_onehot[label]=1
    #TODO tensorize
    for i in range(label.size(0)):
        for j in range(label.size(1)):
            label_onehot[i,j,label[i,j]]=1
    return label_onehot.to(label.device)
def correct_pred(pred,label):
    #Get optimal alignment
    #use DTW
    # introduce blanks at front, back, and inbetween chars
    label_with_blanks = torch.LongTensor(label.size(0)*2+1, label.size(1)).zero_()
    label_with_blanks[1::2]=label.cpu()
    pred_use = pred.cpu().detach()

    batch_size=pred_use.size(1)
    label_len=label_with_blanks.size(0)
    pred_len=pred_use.size(0)

    dtw = torch.FloatTensor(pred_len+1,label_len+1,batch_size).fill_(float('inf'))
    dtw[0,0]=0
    w = max(pred_len//2, abs(pred_len-label_len))
    for i in range(1,pred_len+1):
        dtw[i,max(1, i-w):min(label_len, i+w)+1]=0
    history = torch.IntTensor(pred_len,label_len,batch_size)
    for i in range(1,pred_len+1):
        for j in range(max(1, i-w), min(label_len, i+w)+1):
            cost = 1-pred_use[i-1,torch.arange(0,batch_size).long(),label_with_blanks[j-1,:]]
            per_batch_min, history[i-1,j-1] = torch.min( torch.stack( (dtw[i-1,j],dtw[i-1,j-1],dtw[i,j-1]) ), dim=0)
            dtw[i,j] = cost + per_batch_min
    new_labels = []
    maxlen = 0
    for b in range(batch_size):
        new_label = []
        i=pred_len-1
        j=label_len-1
        #accum += allCosts[b,i,j]
        new_label.append(label_with_blanks[j,b])
        while(i>0 or j>0):
            if history[i,j,b]==0:
                i-=1
            elif history[i,j,b]==1:
                i-=1
                j-=1
            elif history[i,j,b]==2:
                j-=1
            #accum+=allCosts[b,i,j]
            new_label.append(label_with_blanks[j,b])
        new_label.reverse()
        maxlen = max(maxlen,len(new_label))
        new_label = torch.stack(new_label,dim=0)
        new_labels.append(new_label)

    new_labels = [ F.pad(l,(0,maxlen-l.size(0)),value=0) for l in new_labels]
    new_label = torch.LongTensor(maxlen,batch_size)
    for b,l in enumerate(new_labels):
        new_label[:l.size(0),b]=l

    #set to one hot at alignment
    #fuzzy other neighbor preds
    #TODO

    return new_label.to(label.device)

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Interactive script to generate images from trained model')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to training snapshot (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn"')
    parser.add_argument('-r', '--run', default=None, type=str,
                        help='command to run')
    parser.add_argument('-s', '--style_loc', default=None, type=str,
                        help='location of pkl of styles, generated with get_styles.py')

    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)

    config = None
    if args.checkpoint is None and args.config is None:
        print('Must provide checkpoint (with -c)')
        exit()

    if args.run is not None:
        s = args.run.split(',')
        arguments={}
        for pair in s:
            ss = pair.split('=')
            arguments[ss[0]]=ss[1]
    else:
        arguments=None
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.savedir, gpu=args.gpu,  config=args.config, addToConfig=addtoconfig, test =args.test,arguments=arguments, style_loc=args.style_loc)
    else:
        main(args.checkpoint, args.savedir, gpu=args.gpu,  config=args.config, addToConfig=addtoconfig, test =args.test,arguments=arguments, style_loc=args.style_loc)
