from .base import BaseDatasetWrapper, SingleGraphWrapper
from .amazon import AmazonPhotos
from .email import EmailEuCore
from .dblp import DBLP

__all__ = [
    "BaseDatasetWrapper",
    "SingleGraphWrapper",
    "AmazonPhotos",
    "EmailEuCore",
    "DBLP",
]