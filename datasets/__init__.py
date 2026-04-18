from .base import BaseDatasetWrapper
from .amazon import AmazonPhotos
from .email import EmailEuCore
from .dblp import DBLP

__all__ = [
    "BaseDatasetWrapper",
    "AmazonPhotos",
    "EmailEuCore",
    "DBLP",
]