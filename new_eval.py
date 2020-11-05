import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from model.loss import *
from logger import Logger
from trainer import *
from data_loader import getDataLoader
from evaluators import *
import math
from collections import defaultdict
import pickle, csv
#import requests

def update_status(name,message):
    try:
        r = requests.get('http://sensei-status.herokuapp.com/sensei-update/{}?message={}'.format(name,message))
    except requests.exceptions.ConnectionError:
        pass

#from datasets.forms_detect import FormsDetect
#from datasets import forms_detect

logging.basicConfig(level=logging.INFO, format='')

def save_style(location,volume,styles,authors,ids=None,doIds=False, spaced=None,strings=None,doSpaced=False):
    #styles = np.concatenate(styles,axis=0)
    styles = torch.cat(styles,dim=0)
    styles = styles.numpy()
    todump = {'styles':styles, 'authors':authors}

    if doIds:
        todump['ids'] = ids
    if doSpaced:
        todump['spaced'] = spaced#np.concatenate(spaced,axis=0)
        todump['strings'] = strings
    if len(styles)>0:
        authors = np.array(authors)
        loc = location+'.{}'.format(volume)
        pickle.dump(todump, open(loc,'wb'))
        print('saved '+loc)




def main(resume,saveDir,numberOfImages,index,gpu=None, shuffle=False, setBatch=None, config=None, thresh=None, addToConfig=None, test=False, toEval=None, verbosity=2):
    np.random.seed(1234)
    torch.manual_seed(1234)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded iteration {}'.format(checkpoint['iteration']))
        loaded_iteration = checkpoint['iteration']
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        for key in config.keys():
            if type(config[key]) is dict:
                for key2 in config[key].keys():
                    if key2.startswith('pretrained'):
                        config[key][key2]=None
    else:
        checkpoint = None
        config = json.load(open(config))
        loaded_iteration = None
    config['optimizer_type']="none"
    config['trainer']['use_learning_schedule']=False
    config['trainer']['swa']=False
    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    if thresh is not None:
        config['THRESH'] = thresh
        print('Threshold at {}'.format(thresh))
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
            elif value[0]=='[' and value[-1]==']':
                value = value[1:-1].split('-')
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

        
    #config['data_loader']['batch_size']=math.ceil(config['data_loader']['batch_size']/2)
    if 'save_spaced' in config:
        spaced={}
        spaced_val={}
        if toEval is None:
            toEval=['spaced_label']
        elif 'spaced_label' not in toEval:
            toEval.append('spaced_label')
        config['data_loader']['batch_size']=1
        config['validation']['batch_size']=1
        if 'a_batch_size' in config['data_loader']:
            config['data_loader']['a_batch_size']=1
        if 'a_batch_size' in config['validation']:
            config['validation']['a_batch_size']=1
    
    config['data_loader']['shuffle']=shuffle
    #config['data_loader']['rot']=False
    config['validation']['shuffle']=shuffle
    config['data_loader']['eval']=True
    config['validation']['eval']=True
    #config['validation']

    if config['data_loader']['data_set_name']=='FormsDetect':
        config['data_loader']['batch_size']=1
        del config['data_loader']["crop_params"]
        config['data_loader']["rescale_range"]= config['validation']["rescale_range"]

    #print(config['data_loader'])
    if setBatch is not None:
        config['data_loader']['batch_size']=setBatch
        config['validation']['batch_size']=setBatch
    batchSize = config['data_loader']['batch_size']
    if 'batch_size' in config['validation']:
        vBatchSize = config['validation']['batch_size']
    else:
        vBatchSize = batchSize
    if not test:
        data_loader, valid_data_loader = getDataLoader(config,'train')
    else:
        valid_data_loader, data_loader = getDataLoader(config,'test')

    if addDATASET:
        config['DATASET']=valid_data_loader.dataset
    #ttt=FormsDetect(dirPath='/home/ubuntu/brian/data/forms',split='train',config={'crop_to_page':False,'rescale_range':[450,800],'crop_params':{"crop_size":512},'no_blanks':True, "only_types": ["text_start_gt"], 'cache_resized_images': True})
    #data_loader = torch.utils.data.DataLoader(ttt, batch_size=16, shuffle=False, num_workers=5, collate_fn=forms_detect.collate)
    #valid_data_loader = data_loader.split_validation()

    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = eval(config['arch'])(config['model'])
            if config['trainer']['class']=='HWRWithSynthTrainer':
                model = model.hwr
            if 'style' in config['model'] and 'lookup' in config['model']['style']:
                model.style_extractor.add_authors(data_loader.dataset.authors) ##HERE
            ##HACK fix
            keys = list(checkpoint['state_dict'].keys())
            for key in keys:
                if 'style_from_normal' in key: #HACK
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    else:
        model = eval(config['arch'])(config['model'])
    model.eval()
    if verbosity>1:
        model.summary()

    if type(config['loss'])==dict: 
        loss={}#[eval(l) for l in config['loss']]
        for name,l in config['loss'].items():
            loss[name]=eval(l)
    else:   
        loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]


    train_logger = Logger()
    trainerClass = eval(config['trainer']['class'])
    trainer = trainerClass(model, loss, metrics,
                      resume=False, #path
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)
    #saveFunc = eval(trainer_class+'_printer')
    saveFunc = eval(config['data_loader']['data_set_name']+'_eval')

    step=5

    #numberOfImages = numberOfImages//config['data_loader']['batch_size']
    #print(len(data_loader))
    if data_loader is not None:
        train_iter = iter(data_loader)
    if valid_data_loader is not None:
        valid_iter = iter(valid_data_loader)

    with torch.no_grad():

        if index is None:


            if saveDir is not None:
                trainDir = os.path.join(saveDir,'train_'+config['name'])
                validDir = os.path.join(saveDir,'valid_'+config['name'])
                if not os.path.isdir(trainDir):
                    os.mkdir(trainDir)
                if not os.path.isdir(validDir):
                    os.mkdir(validDir)

                if loaded_iteration is not None:
                    with open(os.path.join(validDir,'z_iter_{}.txt'.format(loaded_iteration)),'w') as f:
                        f.write('{}'.format(loaded_iteration))
            else:
                trainDir=None
                validDir=None

            val_metrics_sum = np.zeros(len(metrics))
            val_metrics_list = defaultdict(lambda: defaultdict(list))
            val_comb_metrics = defaultdict(list)

            #if numberOfImages==0:
            #    for i in range(len(valid_data_loader)):
            #        print('valid batch index: {}\{} (not save)'.format(i,len(valid_data_loader)),end='\r')
            #        instance=valid_iter.next()
            #        metricsO,_ = saveFunc(config,instance,model,gpu,metrics)

            #        if type(metricsO) == dict:
            #            for typ,typeLists in metricsO.items():
            #                if type(typeLists) == dict:
            #                    for name,lst in typeLists.items():
            #                        val_metrics_list[typ][name]+=lst
            #                        val_comb_metrics[typ]+=lst
            #                else:
            #                    if type(typeLists) is float or type(typeLists) is int:
            #                        typeLists = [typeLists]
            #                    val_comb_metrics[typ]+=typeLists
            #        else:
            #            val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
            #else:

            ####
            if 'save_spaced' in config:
                spaced={}
                spaced_val={}
                assert(config['data_loader']['batch_size']==1)
                assert(config['validation']['batch_size']==1)
                if 'a_batch_size' in config['data_loader']:
                    assert(config['data_loader']['a_batch_size']==1)
                if 'a_batch_size' in config['validation']:
                    assert(config['validation']['a_batch_size']==1)
            if 'save_nns' in config:
                nns=[]
            if 'save_style' in config:
                if toEval is None:
                    toEval =[]
                if 'style' not in toEval:
                    toEval.append('style')
                if 'author' not in toEval:
                    toEval.append('author')
                styles=[]
                authors=[]
                strings=[]
                stylesVal=[]
                authorsVal=[]
                spacedVal=[]
                stringsVal=[]
                spaced=[]
                
                doIds = config['data_loader']['data_set_name']=='StyleWordDataset'
                #doSpaced = not doIds#?
                doSpaced = 'doSpaced' in config
                if doSpaced:
                    if 'spaced_label' not in toEval:
                        toEval.append('spaced_label')
                    if 'gt' not in toEval:
                        toEval.append('gt')
                ids=[]
                idsVal=[]
                saveStyleEvery=config['saveStyleEvery'] if 'saveStyleEvery' in config else 5000
                saveStyleLoc = config['save_style']
                lastSlash = saveStyleLoc.rfind('/')
                if lastSlash>=0:
                    saveStyleValLoc = saveStyleLoc[:lastSlash+1]+'val_'+saveStyleLoc[lastSlash+1:]
                else:
                    saveStyleValLoc = 'val_'+saveStyleLoc

            if 'save_preds' in config:
                to_save=[]

            validName='valid' if not test else 'test'

            startBatch = config['startBatch'] if 'startBatch' in config else 0
            numberOfBatches = numberOfImages//batchSize
            if numberOfBatches==0 and numberOfImages>1:
                numberOfBatches=1

            #for index in range(startIndex,numberOfImages,step*batchSize):
            batch = startBatch
            numberOfBatches = min(numberOfBatches,max(len(valid_data_loader),len(data_loader)))
            for batch in range(startBatch,numberOfBatches):
            
                #for validIndex in range(index,index+step*vBatchSize, vBatchSize):
                #for validBatch
                    #if valyypidIndex/vBatchSize < len(valid_data_loader):
                if valid_data_loader is not None and batch < len(valid_data_loader) and 'skip_valid' not in config:
                        print('{} batch index: {}/{}       '.format(validName,batch,len(valid_data_loader)),end='\r')
                        #data, target = valid_iter.next() #valid_data_loader[validIndex]
                        #dataT  = _to_tensor(gpu,data)
                        #output = model(dataT)
                        #data = data.cpu().data.numpy()
                        #output = output.cpu().data.numpy()
                        #target = target.data.numpy()
                        #metricsO = _eval_metrics_ind(metrics,output, target)
                        metricsO,aux = saveFunc(config,valid_iter.next(),trainer,metrics,validDir,batch*vBatchSize,toEval=toEval)
                        if type(metricsO) == dict:
                            for typ,typeLists in metricsO.items():
                                if type(typeLists) == dict:
                                    for name,lst in typeLists.items():
                                        val_metrics_list[typ][name]+=lst
                                        val_comb_metrics[typ]+=lst
                                else:
                                    if type(typeLists) is float or type(typeLists) is int:
                                        typeLists = [typeLists]
                                    val_comb_metrics[typ]+=typeLists
                        else:
                            val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
                        if 'save_spaced' in config:
                            spaced_val[aux['name'][0]] = aux['spaced_label'].cpu()
                        if 'save_style' in config:
                            stylesVal.append(aux['style'].cpu())
                            authorsVal+=aux['author']
                            if doIds:
                                idsVal+=aux['name']
                            elif doSpaced:
                                #spacedVal.append(aux[2])
                                spacedVal+=aux['spaced_label']
                                stringsVal+=aux['gt']
                            if batch>0 and batch%saveStyleEvery==0:
                                save_style(saveStyleValLoc,batch,stylesVal,authorsVal,idsVal,doIds, spacedVal,stringsVal, doSpaced)
                                stylesVal=[]
                                authorsVal=[]
                                idsVal=[]
                                spacedVal=[]
                                stringsVal=[]
                            
                    
                if not test and 'skip_train' not in config:
                    #for trainIndex in range(index,index+step*batchSize, batchSize):
                    #    if trainIndex/batchSize < len(data_loader):
                    if batch < len(data_loader):
                            print('train batch index: {}/{}        '.format(batch,len(data_loader)),end='\r')
                            #data, target = train_iter.next() #data_loader[trainIndex]
                            #dataT = _to_tensor(gpu,data)
                            #output = model(dataT)
                            #data = data.cpu().data.numpy()
                            #output = output.cpu().data.numpy()
                            #target = target.data.numpy()
                            #metricsO = _eval_metrics_ind(metrics,output, target)
                            instance = train_iter.next()
                            _,aux=saveFunc(config,instance,trainer,metrics,trainDir,batch*batchSize,toEval=toEval)
                            if 'save_nns' in config:
                                nns+=aux[-1]
                            if 'save_spaced' in config:
                                spaced[aux['name'][0]] = aux['spaced_label'].cpu()
                            if 'save_style' in config:
                                styles.append(aux['style'].cpu())
                                authors+=aux['author']
                                if doIds:
                                    ids+=aux['name']
                                elif doSpaced:
                                    #spaced.append(aux[2])
                                    spaced+=aux['spaced_label']
                                    strings+=aux['gt']
                                if batch>0 and batch%saveStyleEvery==0:
                                    save_style(saveStyleLoc,batch,styles,authors,ids,doIds,spaced,strings,doSpaced)
                                    styles=[]
                                    authors=[]
                                    ids=[]
                                    spaced=[]
                                    strings=[]
                            if 'save_preds' in config:
                                for b in range(batchSize):
                                    try:
                                        to_save.append([instance['name'][b],instance['gt'][b],aux['pred_str'][b],aux['cer'][b]])
                                    except IndexError:
                                        pass

            #if gpu is not None or numberOfImages==0:
            if 'save_preds' in config:
                with open(config['save_preds'],'w') as f:
                    csvwriter=csv.writer(f, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for l in to_save:
                        csvwriter.writerow(l)
                print('wrote results to {}'.format(config['save_preds']))
            if saveDir is None:
                try:
                    if valid_data_loader is not None:
                        for vi in range(batch,len(valid_data_loader)):
                            #print('{} batch index: {}\{} (not save)   '.format(validName,vi,len(valid_data_loader)),end='\r')
                            instance = valid_iter.next()
                            metricsO,aux = saveFunc(config,instance,trainer,metrics,toEval=toEval)
                            if type(metricsO) == dict:
                                for typ,typeLists in metricsO.items():
                                    if type(typeLists) == dict:
                                        for name,lst in typeLists.items():
                                            val_metrics_list[typ][name]+=lst
                                            val_comb_metrics[typ]+=lst
                                    else:
                                        if type(typeLists) is float or type(typeLists) is int:
                                            typeLists = [typeLists]
                                        val_comb_metrics[typ]+=typeLists
                            else:
                                val_metrics_sum += metricsO.sum(axis=0)/metricsO.shape[0]
                            if 'save_spaced' in config:
                                spaced_val[aux['name'][0]] = aux['spaced_label'].cpu()
                            if 'save_style' in config:
                                stylesVal.append(aux['style'].cpu())
                                authorsVal+=aux['author']
                                if doIds:
                                    idsVal+=aux['name']
                                elif doSpaced:
                                    #spacedVal.append(aux[2])
                                    spacedVal+=aux['spaced_label']
                                    stringsVal+=aux['gt']
                                if vi>0 and vi%saveStyleEvery==0:
                                    save_style(saveStyleValLoc,vi,stylesVal,authorsVal,idsVal,doIds,spacedVal,stringsVal,doSpaced)
                                    stylesVal=[]
                                    authorsVal=[]
                                    idsVal=[]
                                    spacedVal=[]
                                    stringsVal=[]
                except StopIteration:
                    print('ERROR: ran out of valid batches early. Expected {} more'.format(len(valid_data_loader)-vi))
            ####
            if valid_data_loader is not None:   
                val_metrics_sum /= len(valid_data_loader)
            print('{} metrics'.format(validName))
            for i in range(len(metrics)):
                print(metrics[i].__name__ + ': '+str(val_metrics_sum[i]))
            for typ in val_comb_metrics:
                print('{} overall mean: {}, std {}'.format(typ,np.mean(val_comb_metrics[typ],axis=0), np.std(val_comb_metrics[typ],axis=0)))
                for name, typeLists in val_metrics_list[typ].items():
                    print('{} {} mean: {}, std {}'.format(typ,name,np.mean(typeLists,axis=0),np.std(typeLists,axis=0)))

            if 'save_nns' in config:
                pickle.dump(nns,open(config['save_nns'],'wb'))
            if 'save_spaced' in config:
                #import pdb;pdb.set_trace()
                #spaced = torch.cat(spaced,dim=1).numpy()
                #spaced_val = torch.cat(spaced_val,dim=1).numpy()
                saveSpacedLoc = config['save_spaced']
                lastSlash = saveSpacedLoc.rfind('/')
                if lastSlash>=0:
                    saveSpacedValLoc = saveSpacedLoc[:lastSlash+1]+'val_'+saveSpacedLoc[lastSlash+1:]
                else:
                    saveSpacedValLoc = 'val_'+saveSpacedLoc
                with open(saveSpacedLoc,'wb') as f:
                    pickle.dump(spaced,f)

                with open(saveSpacedValLoc,'wb') as f:
                    pickle.dump(spaced_val,f)

            if 'save_style' in config:
                if len(styles)>0:
                    assert(not doSpaced)
                    save_style(saveStyleLoc,len(data_loader),styles,authors,ids,doIds)
                if len(stylesVal)>0:
                    save_style(saveStyleValLoc,len(valid_data_loader),stylesVal,authorsVal,idsVal,doIds)
        elif type(index)==int:
            if index>0:
                instances = train_iter
            else:
                index*=-1
                instances = valid_iter
            batchIndex = index//batchSize
            inBatchIndex = index%batchSize
            for i in range(batchIndex+1):
                instance= instances.next()
            #data, target = data[inBatchIndex:inBatchIndex+1], target[inBatchIndex:inBatchIndex+1]
            #dataT = _to_tensor(gpu,data)
            #output = model(dataT)
            #data = data.cpu().data.numpy()
            #output = output.cpu().data.numpy()
            #target = target.data.numpy()
            #print (output.shape)
            #print ((output.min(), output.amin()))
            #print (target.shape)
            #print ((target.amin(), target.amin()))
            #metricsO = _eval_metrics_ind(metrics,output, target)
            saveFunc(config,instance,model,gpu,metrics,saveDir,batchIndex*batchSize,toEval=toEval)
        else:
            for instance in data_loader:
                if index in instance['imgName']:
                    break
            if index not in instance['imgName']:
                for instance in valid_data_loader:
                    if index in instance['imgName']:
                        break
            if index in instance['imgName']:
                saveFunc(config,instance,model,gpu,metrics,saveDir,0,toEval=toEval)
            else:
                print('{} not found! (on {})'.format(index,instance['imgName']))
                print('{} not found! (on {})'.format(index,instance['imgName']))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-i', '--index', default=None, type=int,
                        help='index on instance to process (default: None)')
    parser.add_argument('-n', '--number', default=0, type=int,
                        help='number of images to save out (from each train and valid) (default: 0)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-b', '--batchsize', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-v', '--verbosity', default=2, type=int,
                        help='0,1,2')
    parser.add_argument('-s', '--shuffle', default=False, type=bool,
                        help='shuffle data')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-m', '--imgname', default=None, type=str,
                        help='specify image')
    parser.add_argument('-t', '--thresh', default=None, type=float,
                        help='Confidence threshold for detections')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn".  You can nest keys with k1=k2=k3=v')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set')
    parser.add_argument('-N', '--notify', default='', type=str,
                        help='send messages to server, name')
    parser.add_argument('-e', '--eval', default=None, type=str,
            help='what to evaluate (print) list: "pred"=hwr prediction, "recon"=reconstruction using predicted mask, "recon_gt_mask"=reconstruction using GT mask, "mask"=generated mask for reconstruction "gen"=image generated from interpolated styles, "gen_mask"=mask generated for generated image')
    #parser.add_argument('-E', '--special_eval', default=None, type=str,
    #                    help='what to evaluate (print)')

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

    index = args.index
    if args.index is not None and args.imgname is not None:
        print("Cannot index by number and name at same time.")
        exit()
    if args.index is None and args.imgname is not None:
        index = args.imgname
    if len(args.notify)>0:
        name = args.notify
        update_status(name,'started')
    #toEval = args.spcial_eval if args.spcial_eval is not None else args.eval
    if args.eval is not None and args.eval[0]=='[':
        assert(args.eval[-1]==']')
        toEval=args.eval[1:-1].split(',')
    else:
        toEval=args.eval

    try:
        if args.gpu is not None:
            with torch.cuda.device(args.gpu):
                main(args.checkpoint, args.savedir, args.number, index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, thresh=args.thresh, addToConfig=addtoconfig,test=args.test,toEval=toEval,verbosity=args.verbosity)
        else:
            main(args.checkpoint, args.savedir, args.number, index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, thresh=args.thresh, addToConfig=addtoconfig,test=args.test,toEval=toEval,verbosity=args.verbosity)
    except Exception as er:
        if len(args.notify)>0:
            update_status(name,er)
        raise er
    else:
        if len(args.notify)>0:
            update_status(name,'DONE!')
