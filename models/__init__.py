from .gcn import GCNWrapper
from .gat import GATWrapper
from .sage import SAGEWrapper
from .ppnp import PPNPWrapper, APPNPWrapper
from .mlp import MLPWrapper

__all__ = [
    "GCNWrapper",
    "GATWrapper",
    "SAGEWrapper",
    "PPNPWrapper",
    "APPNPWrapper",
    "MLPWrapper",
]