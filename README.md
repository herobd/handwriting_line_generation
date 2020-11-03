# Handwriting... with style!
This was an Adobe Research intern project performed by [Brian Davis](https://scholar.google.com/citations?user=jXDpalIAAAAJ&hl=en) that resulted in [Text and Style Conditioned GAN for Generation of Offline Handwriting Lines](https://www.bmvc2020-conference.com/assets/papers/0815.pdf) published at [BMVC 2020](https://www.bmvc2020-conference.com/).  Since the end of the internship, Brian has greatly improved upon the method, and the [official code repo](https://github.com/herobd/hw_with_style) for the paper is hosted separately.

The goal of this project is to extract a style vector from handwriting example image(s) (lines), and then, conditioned on that style vector and some given text, generate new handwriting images.

## Requirements
* Python 3.x
* PyTorch 1.1 (greater may have some errors that need fixed still)


To use deformable convs you'll need to copy this repo in the main directory: https://github.com/open-mmlab/mmdetection
Install using this command in the mmdetection directory: `python setup.py develop`
This code only compiles on a GPU version Pytorch, CPU-only does not work.


### Using generate.py

Usage: `python generate.py -c path/to/snapshot.pth -l list_file.json`

The list file has the following format:

It is a list of "jobs":

```
[
    {
        "style": ["path/image1", "path/image1",...],
        "text": ["generate this text", "and this text", ...],
        "out": ["path/save/this", "path/save/that", ...]
    },
    {
        another one...
    }
]
```

All of the style images are appended together to extrac a single style and it is used to generate the given texts.

You can do a single image using the flags `-s "path/image1 path/image2" -t "generate this text" -o path/save/this`.

If you want to interpolate, add `"interpolate":0.1"` to the job. The float is the interpolation step size. It must have only one text (and out path) and each style image will have a style extracted from it individually. The interpolation will be between each of the images' styles (and back to the first one).


### Using new_eval.py

Running `python new_eval.py -c path/to/checkpoint.pth` will run the validation set and return the loss and CER

To write images, use these flags: `-d ../out -n 20 -e [recon,recon_gt_mask,mask,gen,gen_mask] -a data_loader=short=1`

`-d` specifies the output directory, `-n` specifies how many images to output
`-e' has which images to write:

* `recon`: reconstruction using predicted mask
* `recon_gt_mask`: reconstruction using GT mask
* `mask`: predicted mask for reconstruction
* `gen`: instance generated using novel style
* 'gen_mask`: mask generated for above instance

`-a` flag allows changing of the config file. use `key=nested_key=value,another_key=value` format. The `data_loader=short=1` above prints less examples per author (more authors)


You can also use this script to save style vectors (for use in umap_styles.py): `-a save_style=path/to/save/style.pkl,saveStyleEvery=5000`
This will only save the validation set is used alone (in `val_style.pkl`). If training set is desired, use `-n 99999999`, or whatever the size of your dataset is.
It saves a `(val_)style.pkl.#` every `saveStyleEvery` instances. These are automatically globbed by `umap_styles.py` if you pass it `(val_)style.pkl`. Make `saveStyleEvery` big if you don't want this.

### Config file format
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

        "data_dir": "/path/to/data/",    
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
        "style_together": true,                             #append images together (by author) to extract style vector
        "use_hwr_pred_for_style": true,                     #use raw HWR pred as input to style extraction (instead of cheating with aligned GT)
        "hwr_without_style":true,                           #Change to true to pass style vector to HWR model
        "slow_param_names": ["keys"],                       #Multiplies weight updates by 0.1 for param names containing any in the list
        "curriculum": {                                     #Training curriculum
            "0": [["auto"],["auto-disc"]],                  #   iteration to start at: list of list, iteration cycle and components/losses for each iteration
            "1000": [["auto", "auto-gen"],["auto-disc"]],
            "80000": [["count","mask","gt_spaced","mask-gen"],["auto-disc","mask-disc"]],
            "100000": [  [1,"count","mask","gt_spaced","mask-gen"],
                        ["auto","auto-gen","count","mask","gt_spaced","mask_gen"],
                        ["auto","auto-mask","auto-gen","count","mask","gt_spaced","mask_gen"],
                        [2,"gen","gen-auto-style"],
                        [2,"disc","mask-disc"],
                        [2,"auto-disc","mask-disc"]]
        },
        "balance_loss": true,                               #use balancing between CTC loss and reconstruction/adversarial loss
        "interpolate_gen_styles": "extra-0.25",             #interpolate instead of pulling styles out of a hat. This tells it to possibly extrapolate by 0.25 of the distance between the styles

	"text_data": "data/lotr.txt",                       #Text file for generation

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

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

You can overwrite th config from the sanpshot with

  ```
  python train.py --config new_config.json --resume path/to/checkpoint
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

