'''
the backbone of the transformer

author
    zhangsihao yang

logs
    2023-09-25
        file created
'''
from typing import Tuple

import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    '''
    positional encoding
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(max_len, d_model)

        # position : [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term : [d_model // 2]
        div_term = torch.exp(
            torch.arange(
                0, d_model, 2
            ).float() * (-np.log(10000.0) / d_model)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # positional_encoding : [1, max_len, d_model]
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, inputs):
        ''' 
        forward function

        inputs:
        -------
        inputs : [batch_size, n, d_model]
            the input tensor
        '''
        inputs = inputs + self.positional_encoding[:, :inputs.shape[1], :]
        return self.dropout(inputs)


class TransformerEncoder(nn.Module):
    '''
    transformer encoder
    '''

    def __init__(
        self,
        in_out_dim: Tuple[int, int] = (4, 4),
        d_model: int = 16,
        nhead: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.in_linear = nn.Linear(in_out_dim[0], d_model)

        self.positional_encoding = PositionalEncoding(
            d_model, dropout=0.0
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model, nhead, batch_first=True
            ),
            num_layers=num_layers
        )

        self.out_linear = nn.Linear(d_model, in_out_dim[1])

    def forward(self, input_tensor: torch.Tensor):
        '''
        inputs:
        ------
        input_tensor : [batch_size, n, input_dim]
            the input tensor
        '''
        # output_tensor : [batch_size, n, d_model]
        output_tensor = self.in_linear(input_tensor)

        output_tensor = self.positional_encoding(output_tensor)

        output_tensor = self.encoder(output_tensor)

        # output_tensor : [batch_size, n, input_dim]
        output_tensor = self.out_linear(output_tensor)

        return output_tensor
