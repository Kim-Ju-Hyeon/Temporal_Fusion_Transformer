from copy import copy
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from .sub_modules import *
from Interpretable_Multi_head_Attention import *

class TFT(nn.Module):
    def __init__(self, config):
        super(TFT, self).__init__()

        self.config = config

        self.batch_size=config.batch_size
        self.static_variables = config.static_variables
        self.encode_length = config.encode_length
        self.time_varying_categoical_variables = config.time_varying_categoical_variables
        self.time_varying_real_variables_encoder = config.time_varying_real_variables_encoder
        self.time_varying_real_variables_decoder = config.time_varying_real_variables_decoder
        self.num_input_series_to_mask = config.num_masked_series
        self.hidden_size = config.lstm_hidden_dimension
        self.lstm_layers = config.lstm_layers
        self.dropout = config.dropout
        self.embedding_dim = config.embedding_dim
        self.attn_heads = config.attn_heads
        self.num_quantiles = config.num_quantiles
        self.output_size = config.output_size
        self.valid_quantiles = config.vailid_quantiles
        self.seq_length = config.seq_length

