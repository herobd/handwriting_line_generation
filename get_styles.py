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
import pickle



def main(resume,saveDir,index,gpu=None, shuffle=False, setBatch=None, config=None, addToConfig=None, test=False, verbosity=2, transform_style=False):
    assert(saveDir is not None)
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

    train_loc = os.path.join(saveDir,'train_styles_{}.pkl'.format(loaded_iteration))
    if not test:
        val_loc = os.path.join(saveDir,'val_styles_{}.pkl'.format(loaded_iteration))
    else:
        val_loc = os.path.join(saveDir,'test_styles_{}.pkl'.format(loaded_iteration))

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

    if data_loader is not None:
        train_iter = iter(data_loader)
    if valid_data_loader is not None:
        valid_iter = iter(valid_data_loader)

    with torch.no_grad():

        if index is None:


            val_metrics_sum = np.zeros(len(metrics))
            val_metrics_list = defaultdict(lambda: defaultdict(list))
            val_comb_metrics = defaultdict(list)


            validName='valid' if not test else 'test'
            charSpec = trainer.model.char_style_dim>0

            train_styles=[]
            train_authors=[]
            if not test:
                for i,instance in enumerate(data_loader):
                    print('train: {}/{}       '.format(i,len(data_loader)),end='\r')
                    image, label = trainer._to_tensor(instance)
                    batch_size = label.size(1)
                    label_lengths = instance['label_lengths']
                    a_batch_size = trainer.a_batch_size if 'a_batch_size' in instance else None

                    style = trainer.model.extract_style(image,label,a_batch_size)

                    if transform_style:
                        style = trainer.model.generator.style_emb(style)
                    if charSpec:
                        for b in range(batch_size):
                            train_styles.append((style[0][b].cpu(),style[1][b].cpu(),style[2][b].cpu()))
                    else:
                        train_styles.append(style.cpu())
                    train_authors+=instance['author']


                    trainer.model.spaced_label=None
                    trainer.model.mask=None
                    trainer.model.gen_mask=None
                    trainer.model.top_and_bottom=None
                    trainer.model.counts=None
                    trainer.model.pred=None
                    trainer.model.spacing_pred=None
                    trainer.model.mask_pred=None
                    trainer.model.gen_spaced=None
                    trainer.model.spaced_style=None
                    trainer.model.mu=None
                    trainer.model.sigma=None
                if charSpec:
                    train_styles = [ (s[0].numpy(),s[1].numpy(),s[2].numpy()) for s in train_styles]
                else:
                    train_styles = torch.cat(train_styles,dim=0).numpy()
                train_authors = np.array(train_authors)
                pickle.dump({'styles':train_styles,'authors':train_authors},open(train_loc,'wb'))
                print('saved {}'.format(train_loc))
                

            val_styles=[]
            val_authors=[]
            for i,instance in enumerate(valid_data_loader):
                print('{}: {}/{}       '.format(validName,i,len(valid_data_loader)),end='\r')
                image, label = trainer._to_tensor(instance)
                batch_size = label.size(1)
                label_lengths = instance['label_lengths']
                a_batch_size = trainer.a_batch_size if 'a_batch_size' in instance else None

                style = trainer.model.extract_style(image,label,a_batch_size)

                if transform_style:
                    style = trainer.model.generator.style_emb(style)
                if charSpec:
                    for b in range(batch_size):
                        val_styles.append((style[0][b].cpu(),style[1][b].cpu(),style[2][b].cpu()))
                else:
                    val_styles.append(style.cpu())

                val_authors+=instance['author']

                trainer.model.spaced_label=None
                trainer.model.mask=None
                trainer.model.gen_mask=None
                trainer.model.top_and_bottom=None
                trainer.model.counts=None
                trainer.model.pred=None
                trainer.model.spacing_pred=None
                trainer.model.mask_pred=None
                trainer.model.gen_spaced=None
                trainer.model.spaced_style=None
                trainer.model.mu=None
                trainer.model.sigma=None
            if charSpec:
                val_styles = [ (s[0].numpy(),s[1].numpy(),s[2].numpy()) for s in val_styles]
                assert(len(val_styles) == len(val_authors))
            else:
                val_styles = torch.cat(val_styles,dim=0).numpy()
            val_authors = np.array(val_authors)
            pickle.dump({'styles':val_styles,'authors':val_authors},open(val_loc,'wb'))
            print('saved {}'.format(val_loc))



if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-i', '--index', default=None, type=int,
                        help='index on instance to process (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu)')
    parser.add_argument('-b', '--batchsize', default=None, type=int,
                        help='Set the batch size (default: use config)')
    parser.add_argument('-v', '--verbosity', default=2, type=int,
                        help='0,1,2')
    parser.add_argument('-s', '--shuffle', default=False, type=bool,
                        help='shuffle data')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-m', '--imgname', default=None, type=str,
                        help='specify image')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn".  You can nest keys with k1=k2=k3=v')
    parser.add_argument('-T', '--test', default=False, action='store_const', const=True,
                        help='Run test set (default is train and valid)')
    parser.add_argument('-S', '--transformstyle', default=False, action='store_const', const=True,
                        help="use generator's style embedding function")
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


    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.savedir, index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, addToConfig=addtoconfig,test=args.test,verbosity=args.verbosity,transform_style=args.transformstyle)
    else:
        main(args.checkpoint, args.savedir, index, gpu=args.gpu, shuffle=args.shuffle, setBatch=args.batchsize, config=args.config, addToConfig=addtoconfig,test=args.test,verbosity=args.verbosity,transform_style=args.transformstyle)
