from copy import copy
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric

from sub_modules import *
from Interpretable_Multi_head_Attention import *

class TFT(nn.Module):
    def __init__(self, config):
        super(TFT, self).__init__()

        # self.config = config
        #
        # self.batch_size=config.batch_size
        # self.static_variables = config.static_variables
        # self.encode_length = config.encode_length
        # self.time_varying_categoical_variables = config.time_varying_categoical_variables
        # self.time_varying_real_variables_encoder = config.time_varying_real_variables_encoder
        # self.time_varying_real_variables_decoder = config.time_varying_real_variables_decoder
        # self.num_input_series_to_mask = config.num_masked_series
        # self.hidden_size = config.lstm_hidden_dimension
        # self.lstm_layers = config.lstm_layers
        # self.dropout = config.dropout
        # self.embedding_dim = config.embedding_dim
        # self.attn_heads = config.attn_heads
        # self.num_quantiles = config.num_quantiles
        # self.output_size = config.output_size
        # self.valid_quantiles = config.vailid_quantiles
        # self.seq_length = config.seq_length

        self.device = config['device']
        self.batch_size=config['batch_size']
        self.static_variables = config['static_variables']
        self.encode_length = config['encode_length']
        self.time_varying_categorical_variables =  config['time_varying_categorical_variables']
        self.time_varying_real_variables_encoder =  config['time_varying_real_variables_encoder']
        self.time_varying_real_variables_decoder =  config['time_varying_real_variables_decoder']
        self.num_input_series_to_mask = config['num_masked_series']
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.embedding_dim = config['embedding_dim']
        self.attn_heads = config['attn_heads']
        self.num_quantiles = config['num_quantiles']
        self.valid_quantiles = config['vailid_quantiles']
        self.seq_length = config['seq_length']

        self.static_embedding_layers = nn.ModuleList()
        for i in range(self.static_variables):
            emb = nn.Embedding(config['static_embedding_vocab_sizes'][i], config['embedding_dim']).to(self.device)
            self.static_embedding_layers.append(emb)

        self.time_varying_embedding_layers = nn.ModuleList()
        for i in range(self.time_varying_categoical_variables):
            emb = TimeDistributed(
                nn.Embedding(config['time_varying_embedding_vocab_sizes'][i], config['embedding_dim']),
                batch_first=True).to(self.device)
            self.time_varying_embedding_layers.append(emb)

        self.time_varying_linear_layers = nn.ModuleList()
        for i in range(self.time_varying_real_variables_encoder):
            emb = TimeDistributed(nn.Linear(1, config['embedding_dim']), batch_first=True).to(self.device)
            self.time_varying_linear_layers.append(emb)

        self.encoder_variable_selection = VariableSelectionNetwork(config['embedding_dim'],
                                                                   (config['time_varying_real_variables_encoder'] +
                                                                    config['time_varying_categoical_variables']),
                                                                   self.hidden_size,
                                                                   self.dropout,
                                                                   config['embedding_dim'] * config['static_variables'])

        self.decoder_variable_selection = VariableSelectionNetwork(config['embedding_dim'],
                                                                   (config['time_varying_real_variables_decoder'] +
                                                                    config['time_varying_categoical_variables']),
                                                                   self.hidden_size,
                                                                   self.dropout,
                                                                   config['embedding_dim'] * config['static_variables'])

        self.lstm_encoder_input_size = config['embedding_dim'] * (config['time_varying_real_variables_encoder'] +
                                                                  config['time_varying_categoical_variables'] +
                                                                  config['static_variables'])

        self.lstm_decoder_input_size = config['embedding_dim'] * (config['time_varying_real_variables_decoder'] +
                                                                  config['time_varying_categoical_variables'] +
                                                                  config['static_variables'])

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_layers,
                                    dropout=config['dropout'])

        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_layers,
                                    dropout=config['dropout'])

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size))

        self.static_enrichment = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                      self.dropout, config['embedding_dim'] * self.static_variables)

        self.position_encoding = PositionalEncoder(self.hidden_size, self.seq_length)

        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))

        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size, self.num_quantiles), batch_first=True)

    def init_hidden(self):
        return torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size, device=self.device)

    def apply_embedding(self, x, static_embedding, apply_masking):
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps
        # Time-varying real embeddings
        if apply_masking:
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_decoder):
                emb = self.time_varying_linear_layers[i + self.num_input_series_to_mask](
                    x[:, :, i + self.num_input_series_to_mask].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else:
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_encoder):
                emb = self.time_varying_linear_layers[i](x[:, :, i].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        ##Time-varying categorical embeddings (ie hour)
        time_varying_categoical_vectors = []
        for i in range(self.time_varying_categoical_variables):
            emb = self.time_varying_embedding_layers[i](
                x[:, :, self.time_varying_real_variables_encoder + i].view(x.size(0), -1, 1).long())
            time_varying_categoical_vectors.append(emb)
        time_varying_categoical_embedding = torch.cat(time_varying_categoical_vectors, dim=2)

        ##repeat static_embedding for all timesteps
        static_embedding = torch.cat(time_varying_categoical_embedding.size(1) * [static_embedding])
        static_embedding = static_embedding.view(time_varying_categoical_embedding.size(0),
                                                 time_varying_categoical_embedding.size(1), -1)

        ##concatenate all embeddings
        embeddings = torch.cat([static_embedding, time_varying_categoical_embedding, time_varying_real_embedding],
                               dim=2)

        return embeddings.view(-1, x.size(0), embeddings.size(2))

    def encode(self, x, hidden=None):

        if hidden is None:
            hidden = self.init_hidden()

        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))

        return output, hidden

    def decode(self, x, hidden=None):

        if hidden is None:
            hidden = self.init_hidden()

        output, (hidden, cell) = self.lstm_decoder(x, (hidden, hidden))

        return output, hidden

    def forward(self, x):

        ##inputs should be in this order
        # static
        # time_varying_categorical
        # time_varying_real

        embedding_vectors = []
        for i in range(self.static_variables):
            # only need static variable from the first timestep
            emb = self.static_embedding_layers[i](x['identifier'][:, 0, i].long().to(self.device))
            embedding_vectors.append(emb)

        ##Embedding and variable selection
        static_embedding = torch.cat(embedding_vectors, dim=1)
        embeddings_encoder = self.apply_embedding(x['inputs'][:, :self.encode_length, :].float().to(self.device),
                                                  static_embedding, apply_masking=False)
        embeddings_decoder = self.apply_embedding(x['inputs'][:, self.encode_length:, :].float().to(self.device),
                                                  static_embedding, apply_masking=True)
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_encoder[:, :, :-(self.embedding_dim * self.static_variables)],
            embeddings_encoder[:, :, -(self.embedding_dim * self.static_variables):])
        embeddings_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_decoder[:, :, :-(self.embedding_dim * self.static_variables)],
            embeddings_decoder[:, :, -(self.embedding_dim * self.static_variables):])

        pe = self.position_encoding(torch.zeros(self.seq_length, 1, embeddings_encoder.size(2)).to(self.device)).to(
            self.device)

        embeddings_encoder = embeddings_encoder + pe[:self.encode_length, :, :]
        embeddings_decoder = embeddings_decoder + pe[self.encode_length:, :, :]

        ##LSTM
        lstm_input = torch.cat([embeddings_encoder, embeddings_decoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output + lstm_input)

        ##static enrichment
        static_embedding = torch.cat(lstm_output.size(0) * [static_embedding]).view(lstm_output.size(0),
                                                                                    lstm_output.size(1), -1)
        attn_input = self.static_enrichment(lstm_output, static_embedding)

        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        # attn_input = self.position_encoding(attn_input)

        ##Attention
        attn_output, attn_output_weights = self.multihead_attn(attn_input[self.encode_length:, :, :],
                                                               attn_input[:self.encode_length, :, :],
                                                               attn_input[:self.encode_length, :, :])

        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[self.encode_length:, :, :]
        attn_output = self.post_attn_norm(attn_output)

        output = self.pos_wise_ff(attn_output)  # [self.encode_length:,:,:])

        ##skip connection over Decoder
        output = self.pre_output_gate(output) + lstm_output[self.encode_length:, :, :]

        # Final output layers
        output = self.pre_output_norm(output)
        output = self.output_layer(output.view(self.batch_size, -1, self.hidden_size))

        return output, encoder_output, decoder_output, attn_output, attn_output_weights, encoder_sparse_weights, decoder_sparse_weights

