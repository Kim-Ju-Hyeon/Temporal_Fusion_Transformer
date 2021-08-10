import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, dropout, hidden_context_size=None,
                 batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout

        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()

        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size),
                                           batch_first=batch_first)

        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size, self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(self.dropout)

        self.norm = TimeDistributed(nn.BatchNorm1d(self.output_size), batch_first=batch_first)

        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.norm(x)

        return


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout)

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(
                self.single_variable_grns[i](embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)

        outputs = var_outputs * sparse_weights

        outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights
