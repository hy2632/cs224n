"""Assortment of layers for use in QANet.

Author:
    Hua Yao (hy2632@columbia.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax



