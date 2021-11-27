import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs=lambda x: torch.log(torch.abs(x))

# three main components of GLOW
# flow, inverse of flow and log-determinants
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        '''
        BN to alleviate the problems encountered when training deep models
        :param in_channel:
        :param LOGDET:
        '''

