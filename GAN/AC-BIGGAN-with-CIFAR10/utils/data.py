from typing import Tuple, Optional, Union
import ssl
import torch
from torch.utils import data
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

ssl._create_default_https_context = ssl._create_unverified_context

_CIFAR_MEAN = (0.485, 0.456, 0.406)
_CIFAR_STD = (0.229, 0.224, 0.225)

_NEG1TO1_MEAN_STD = (0.5, 0.5, 0.5)

# Downloading CIFAR10 Dataset
def get_CIFAR10(
    concatDataset: bool = False,
    preprocessing: Optional[transforms.Compose] = None,
) -> Union[Tuple[data.Dataset, data.Dataset], data.Dataset]:
    # Preprocessing
    if not preprocessing:
        preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(_NEG1TO1_MEAN_STD, _NEG1TO1_MEAN_STD),
            ]
        )

    train_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=preprocessing
    )
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=preprocessing
    )

    return (
        data.ConcatDataset((train_data, test_data))
        if concatDataset
        else (train_data, test_data)
    )


def inverseNormalize(imgs, mean=_CIFAR_MEAN, std=_CIFAR_STD):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv

    return F.normalize(imgs, mean=mean_inv, std= std_inv)
