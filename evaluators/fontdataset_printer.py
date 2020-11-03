# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from .hwdataset_printer import HWDataset_printer


def FontDataset_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    return HWDataset_printer(config,instance, model, gpu, metrics, outDir, startIndex, lossFunc)
