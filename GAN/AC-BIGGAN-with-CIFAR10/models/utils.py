import torch
import torch.nn as nn

def weights_init(m):
    """
    Weight initialisation using orthogonal matrix
    """
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.Embedding):
        nn.init.orthogonal_(m.weight)
    else:
        pass