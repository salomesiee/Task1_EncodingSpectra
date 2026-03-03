import numpy as np
import torch
from torchvision import transforms

from preprocessing import PreprocessingPipeline


class TransformsComposer:
    """
    Composes a sequence of 1D spectrum transforms.
    Each transform operates on a dict with a 'Data' key and returns the same dict.
    """
    def __init__(self, preprocessing_funcs, normalize=True):
        self.preprocessing_pipeline = PreprocessingPipeline(steps=preprocessing_funcs)
        transfs = [TrApplyPreprocessing(self.preprocessing_pipeline)]
        if normalize:
            transfs.append(TrMinMaxNormalize())
        transfs.append(TrAddChannel())
        self.compose = transforms.Compose(transfs)

    def __call__(self, x):
        return self.compose(x) 


class TrApplyPreprocessing:
    def __init__(self, preprocessing_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline

    def __call__(self, x):
        data_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        ret = torch.from_numpy(self.preprocessing_pipeline.process(data_np)).float()
        return ret


class TrMinMaxNormalize:
    """Normalizes a spectrum tensor to [0, 1]."""
    def __call__(self, x):
        ret  = (x - x.min()) / (x.max() - x.min())
        return ret


class TrStandardNormalize:
    """Normalizes a spectrum tensor to zero mean and unit variance."""
    def __call__(self, x):
        ret = (x - x.mean()) / x.std if x.std > 0 else x - x.mean()
        return ret


class TrAddChannel:
    """Adds a channel dimension: (L,) -> (1, L)."""
    def __call__(self, x):
        return x.unsqueeze(0)