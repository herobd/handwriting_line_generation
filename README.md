# Handwriting line generation

UPDATE 3 Nov 2021: I've done a major cleaning of the code. I had initially left in the code to run all the variations of the method I had tried (genetators, discriminators, losses, HWR models, etc) which made the code very hard to read. You can still find these in the `full-messy` branch. But I've cut out almost everything except the parts relevant to the method used for the paper in the master branch.

![examples of generated handwriting](https://i.ibb.co/ChtGqwW/handwriting-for-readme.png)

This is the code for the paper "Text and Style Conditioned GAN for the Generation of Offline-Handwriting Lines" published at BMVC 2020. 

Paper: https://arxiv.org/abs/2009.00678

Video: https://youtu.be/TuYlE8qu_VE

This was originally Brian Davis's summer 2019 internship project at Adobe (https://github.com/adobe-research/hw_with_style). It was then extended afterwards (while at BYU) and finally published.

The trained models (snapshots) are available as a file in the release (https://github.com/herobd/handwriting_line_generation/releases/tag/w1.0).

Code structure based on victoresque pytorch template.

## Requirements
* Python 3.x
* PyTorch >1.0
* torchvision
* opencv
* scikit-image
* editdistance



## Reproducability instructions
In the `configs` directory are several jsons which have the parameters used for the paper. The `"data_loader": "data_dir"` needs set to the location of the dataset directory. You can also adjust the GPU options here.

First the handwriting recognition model and feature encoder networks need to be trained.

HWR: `python train.py -c configs/cf_IAM_hwr_cnnOnly_batchnorm_aug.json`

Encoder: `python train.py -c configs/cf_IAM_auto_2tight_newCTC.json`

Then the generator can be trained: `python train.py -c configs/cf_IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG.json`


The RIMES dataset will use:

HWR: `python train.py -c configs/cf_RIMESLines_hwr_cnnOnly_batchnorm_aug.json`

GAN: `python train.py -c configs/cf_RIMESLinesslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG.json`

RIMES resuses the IAM encoder.

Figures for the paper were made using `generate.py`

## Generating images using generate.py


Usage: `python generate.py -c path/to/snapshot.pth -d output_directory -g #[optional gpu flag] -s style_pickle_file[optional] -T[optional, use test set]

You can also use `-h`

It has several "modes" to select from once it's loaded the model. Use `h` to display them. Most will ask for additional input (desired text, number of styles, etc). The output is saved to the supplied `-d` location.

Probably the two most useful modes are: 
* "from-to", which will ask for two image paths and an output text. It extracts the styles from the given images and then generates a series of images interpolating between the two styles with the given text.
* "Random", will generate images using random styles (interpolated from dataset style pickle). It asks how many samples you want and what text.

Some modes need the full dataset's styles in a pickle. I've included the IAM/RIME test set style pickles with the trained snapshots. Use `get_styles.py` to extract any additional ones.

`python get_styles.py -c path/to/snapshot.pth -d output_directory -g #[optional gpu flag] -T[optional, do test set, otherwise does trian and valid]`

## Data

The RIMES training uses a text file of French (`french_news.txt`). This really can be any medium (maybe even small) French text corpus. I used the one found at https://webhose.io/free-datasets/french-news-articles/
To compile it's jsons into the .txt file, use the script found at `data/compile_french_new.py`. First create a file listing all the jsons (`ls > jsons.txt`) then run the script.

I don't remember where I got the english text corpus (at least part of it is the OANC https://www.anc.org/data/oanc/download/ dataset), but really any one should do. It just expects a txt file with just text.

## Folder Structure
  ```
  
  │
  ├── train.py - Use this to train
  ├── new_eval.py - Use this to evaluate and save images. Some examples of how to run this in notes.txt. This was used to generate the reconstruction images for the paper.
  ├── get_styles.py - This uses a trained model to extract style vectors from a dataset and save them.
  ├── generate.py - This is an interactive script to generate images using a trained model, including interpolations. Figures for the paper were generally created using this.
  ├── umap_styles.py - This generates the umap plots used in the paper
  ├── graph.py - Display plots given a training snapshot
  ├── old_generate.py - This will generate images given a json having lists of style images, text lines, and output paths. (I don't know if this works still)
  ├── eval_writer_id.py - was intended to evaluate writer identification performance given the style vectors, but I don't know if I ever got it working correctly.
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders (unused?)
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data/ - this has various files that were convenient to keep with the project
  ├── data_loader/ 
  │   └── data_loaders.py - This just gets you the right dataset
  │
  ├── datasets/ - default datasets folder
  │   ├── hw_dataset.py - basic dataset made to handle IAM data at the line level
  │   ├── author_hw_dataset.py - This sorts instances by author and has a 'a_batch_size', which is how many by each author should be in the batch (batch_size is number of authors)
  │   ├── author_rimeslines_dataset.py - Same as author_hw_dataset.py, but for RIMES lines
  │   └── test*.py - This are scripts to run through a dataset and simply display what's being returned. For debugging purposes.
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── loss.py - has all losses, here or imported here
  │   ├── cnn_lstm.py - Code from Start, Follow, Read, with minor tweaks to allow Group Norm and logsoftmax at the end
  │   ├── cnn_only_hwr.py - Alternate network to do HWR without recourrent layers. Doesn't work too well.
  │   ├── discriminator_ap.py - Multi-scale patch discriminator using spectral norm
  │   ├── hw_with_style.py - Primary model which contains all submodels.
  │   ├── count_cnn.py - Spacing prediction subnetwork
  │   ├── pure_gen.py - StyleGAN based generator
  │   ├── char_style.py - style extractor which uses character specific heads
  │   └── autoencoder.py - Contains autoencoding models pre-trained for perceptual loss
  │
  ├── saved/ - default checkpoints folder
  │
  ├── configs/ - configuration files for reproducibility
  ├── old_configs/ - all the other config files I've used during development/other projects
  │
  ├── trainer/ - trainers
  │   ├── hw_with_style_trainer.py - This has the code to run training for both HWR model and generative model (specify curriculum for generator training)
  │   └── auto_trainer.py - For training autoencoder used in perceptual loss
  │
  └── utils/
      ├── util.py - importantly has code to create mask from handwriting image and extact centerline from handwriting image
      ├── augmentation.py - Chris's brightness augmentation
      ├── curriculum.py - this object handles tracking the curriculum during training
      ├── error_rates.py - character error, etc
      ├── grid_distortion.py - Curtis's augmentation
      ├── normalize_line.py - functions to noramlize a line image
      ├── parseIAM.py - parses the xmls IAM has
      ├── parseRIMESlines.py - parse the GT for RIMES into line images
      ├── string_utils.py - used for converting string characters to their class numbers and back
      └── util.py - various functions
  ```


## Using old_generate.py

I'm not sure if this still works...

Usage: `python old_generate.py -c path/to/snapshot.pth -l list_file.json`

The list file has the following format:

It is a list of "jobs. Each job looks like

```
{
    "style": ["path/image1", "path/image1",...],
    "text": ["generate this text", "and this text", ...],
    "out": ["path/save/this", "path/save/that", ...]
}
```

All of the style images are appended together to extrac a single style and it is used to generate the given texts.

You can do a single image using the flags `-s "path/image1 path/image2" -t "generate this text" -o path/save/this`.

If you want to interpolate, add `"interpolate:0.1"` to the object. The float is the interpolation step size. It must have only one text (and out path) and each style image will have a style extracted from it individually. The interpolation will be between each of the images' styles (and back to the first one).


## Config file format
Config files are in `.json` format. 
  ```
{
    "name": "long_Gskip",                                       #name for saving
    "cuda": true,                                               #use GPUs
    "gpu": 0,                                                   #only single GPU supported
    "save_mode": "state_dict",                                  #can change to save whole model
    "override": true,                                           #if resuming, replace config file
    "super_computer":false,                                     #whether to mute log output
    "data_loader": {
        "data_set_name": "AuthorHWDataset",                     #class name

        "data_dir": "/trainman-mount/trainman-storage-8308c0a4-7f25-47ad-ae22-1de9e3faf4ad",    #IAM loaction on sensei
        "Xdata_dir": "../data/IAM/",
        "batch_size": 1,
        "a_batch_size": 2,
        "shuffle": true,
        "num_workers": 2,

        "img_height": 64,
        "max_width": 1400,
        "char_file": "./data/IAM_char_set.json",
        "mask_post": ["thresh","dilateCircle","errodeCircle"],
        "mask_random": false,
        "spaced_loc": "../saved/spaced/spaced.pkl"
    },
    "validation": {
        "shuffle": false,
        "batch_size": 3,
        "a_batch_size": 2,
        "spaced_loc": "../saved/spaced/val_spaced.pkl"
    },

    
    "lr_scheduler_type": "none",
 
    "optimizer_type": "Adam",
    "optimizer": {
        "lr": 0.00001,
        "weight_decay": 0,
        "betas": [0.5,0.99]
    },
    "optimizer_type_discriminator": "Adam",                 #seperate optimizer for discriminators
    "optimizer_discriminator": {
        "lr": 0.00001,
        "weight_decay": 0,
        "betas": [0,0.9]
    },
    "loss": {                                               #adversarial losses (generative and discriminator) are hard coded and not specified here
        "auto": "pyramidL1Loss",
        "key": "pushMinDist",
        "count": "MSELoss",
        "mask": "HingeLoss",
        "feature": "L1Loss",
        "reconRecog": "CTCLoss",
        "genRecog": "CTCLoss",
        "genAutoStyle": "L1Loss"
    },
    "loss_weights": {                                       #multiplied to loss to balance them
        "auto": 1,
        "discriminator": 0.1,
        "generator": 0.01,
        "key": 0.001,
        "count": 0.1,
        "mask": 0.1,
        "mask_generator": 0.01,
        "mask_discriminator": 0.01,
        "feature": 0.0000001,
        "reconRecog": 0.000001,
        "genRecog": 0.0001,
        "style_discriminator": 0.1,
        "style_generator": 0.01,
        "genAutoStyle": 0.1

    },
    "loss_params":                                          #additional params passed directly to function
        {
            "auto": {"weights":[0.4,0.3,0.3],
                     "pool": "avg"},
            "key": {"dist":"l1",
                    "thresh": 1.0},
            "mask": {"threshold": 4}
        },
    "metrics": [],                                          #unused
    "trainer": {
        "class": "HWWithStyleTrainer",
        "iterations": 700000,                               #Everything is iterations, because epochs are weird.
        "save_dir": "../saved/",
        "val_step": 2000,                                   #how frequently to run through validation set
        "save_step": 5000,                                  #how frequently to save a seperate snapshot of the training & model
        "save_step_minor": 250,                             #how frequently to save a "latest" model (overwrites)
        "log_step": 100,                                    #how frequently to print training stats
        "verbosity": 1,
        "monitor": "loss",
        "monitor_mode": "none",
        "space_input": true,
        "style_together": true,                             #append sty
        "use_hwr_pred_for_style": true,
        "hwr_without_style":true,
        "slow_param_names": ["keys"],
        "curriculum": {
            "0": [["auto"],["auto-disc"]],
            "1000": [["auto", "auto-gen"],["auto-disc"]],
            "80000": [["count","mask","gt_spaced","mask-gen"],["auto-disc","mask-disc"]],
            "100000": [  [1,"count","mask","gt_spaced","mask-gen"],
                        ["auto","auto-gen","count","mask","gt_spaced","mask_gen"],
                        ["auto","auto-mask","auto-gen","count","mask","gt_spaced","mask_gen"],
                        [2,"gen","gen-auto-style"],
                        [2,"disc","mask-disc"],
                        [2,"auto-disc","mask-disc"]]
        },
        "balance_loss": true,
        "interpolate_gen_styles": "extra-0.25",

	"text_data": "data/lotr.txt",

        "use_learning_schedule": false
    },
    "arch": "HWWithStyle", 
    "model": {
        "num_class": 80,
        "generator": "SpacedWithMask",
        "gen_dim": 128,
        "gen_n_res1": 2,
        "gen_n_res2": 3,
        "gen_n_res3": 2,
        "gen_use_skips": true,
	"hwr": "CRNN_group_norm_softmax",
        "pretrained_hwr": "../saved/IAM_hwr_softmax_aug/checkpoint-latest.pth",
        "hwr_frozen": true,
        "style": "new",
        "style_norm":"group",
        "style_activ":"relu",
        "style_dim": 256,
        "num_keys": 64,
        "global_pool": true,
        "discriminator": "two-scale-better more low global",
        "spacer": "duplicates",
        "create_mask": true,
        "mask_discriminator": "with derivitive"

    }
}
  ```

##  Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

##  Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```


The checkpoints will be saved in `save_dir/name/`.

The config file is saved in the same folder. (as a reference only, the config is loaded from the checkpoint)

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
  }
  ```


## UMAPing

Save styles using `get_styles.py`
Then `umap_styles.py styles.pkl [image dir]`

