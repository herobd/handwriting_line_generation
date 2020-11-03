# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
from model.unet import UNet
from model.unet_dilated import UNetDilated
#from model.sol_eol_finder import SOL_EOL_Finder
#from model.detector import Detector
#from model.line_follower import LineFollower

#from evaluators.formsdetect_printer import FormsDetect_printer
#from evaluators.formsboxdetect_printer import FormsBoxDetect_printer
#from evaluators.formsboxpair_printer import FormsBoxPair_printer
#from evaluators.formsgraphpair_printer import FormsGraphPair_printer
#from evaluators.formsfeaturepair_printer import FormsFeaturePair_printer
#from evaluators.formslf_printer import FormsLF_printer
#from evaluators.formspair_printer import FormsPair_printer
#from evaluators.ai2d_printer import AI2D_printer
#from evaluators.randommessages_printer import RandomMessagesDataset_printer
#from evaluators.randomdiffusion_printer import RandomDiffusionDataset_printer
from .hwdataset_printer import HWDataset_printer
from .fontdataset_printer import FontDataset_printer
from .hwdataset_eval import HWDataset_eval
from .styleworddataset_eval import StyleWordDataset_eval
from .dupdataset_eval import *#FontDataset_eval, SameFontDataset_eval, AuthorHWDataset_eval, AuthorWordDataset_eval

#def FormsPair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None):
#    return AI2D_printer(config,instance, model, gpu, metrics, outDir, startIndex)
