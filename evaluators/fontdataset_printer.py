
from .hwdataset_printer import HWDataset_printer


def FontDataset_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None, lossFunc=None):
    return HWDataset_printer(config,instance, model, gpu, metrics, outDir, startIndex, lossFunc)
