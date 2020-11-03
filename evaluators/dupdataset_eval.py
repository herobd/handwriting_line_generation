# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from .hwdataset_eval import HWDataset_eval


def FontDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return HWDataset_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc,toEval)
def SameFontDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return HWDataset_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc,toEval)
def AuthorHWDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return HWDataset_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc,toEval)
def MixedAuthorHWDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return HWDataset_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc,toEval)
def AuthorWordDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return HWDataset_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc,toEval)
def MixedAuthorWordDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return HWDataset_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc,toEval)
