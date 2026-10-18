"""Standalone copy of the original AutoDecoderPointsetsV2cmp architecture."""
from typing import List
import torch
from torch import nn


class GaussianPositionalEncoding(nn.Module):
    def __init__(self, input_dim : int, mapped_dim : int, scale : float):
        super(GaussianPositionalEncoding, self).__init__()
        # create the random Gaussian Matrix
        self.B = nn.Parameter(torch.randn((input_dim, mapped_dim), dtype=torch.float32) * scale,
                    requires_grad=False)    # no need to update

    def forward(self, x, mask = None):
        if mask is not None:
            x = x * mask
        x_proj = (2. * 3.1415926536 * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)




class ConcatLinear(nn.Module):
    def __init__(self, inputdim: int, outputdim: int):
        super(ConcatLinear, self).__init__()
        self.linear = nn.Linear(inputdim, outputdim)

    def forward(self, x, condition):
        input = torch.cat((x, condition), dim=-1)
        return self.linear(input)


class AutoDecoderPointsetsV2cmp(nn.Module):
    def __init__(self, pointset_featuredim : int, hidden_sizes : List[int], output_dim : int, pointset_size : int, scale : float, mapped_dim : int = 8):
        super(AutoDecoderPointsetsV2cmp, self).__init__()

        assert len(hidden_sizes) >= 1

        self.encoding_layer = GaussianPositionalEncoding(pointset_featuredim, mapped_dim, scale)
        self.network = nn.Sequential()

        concatlength = 2 * mapped_dim
        self.network.add_module("StartLayer", nn.Linear(concatlength, hidden_sizes[0]))
        prev_dim = hidden_sizes[0]

        for (i, hidden_dim) in enumerate(hidden_sizes):
            if i == 0:
                continue
            self.network.add_module("HiddenLayer_%d" % i, ConcatLinear(prev_dim + concatlength, hidden_dim))
            self.network.add_module("ReLU_%d" % i, nn.ReLU())
            prev_dim = hidden_dim

        # add last layer
        self.network.add_module("OutputLayer", ConcatLinear(prev_dim + concatlength, pointset_size * output_dim))

        self.pointset_size = pointset_size
        self.output_dim = output_dim

    def forward(self, pointset: torch.Tensor, distribution_feature: torch.Tensor) -> torch.Tensor:
        concatlatent = torch.cat((pointset, distribution_feature), dim=-1)
        modelinput = self.encoding_layer(concatlatent)
        y = self.network[0](modelinput)
        for i in range(1, len(self.network) - 1, 2):
            y = self.network[i+1](self.network[i](y, modelinput))
        y = self.network[-1](y, modelinput)
        if len(y.shape) == 2:           # means we have batch
            return torch.reshape(y, (y.shape[0], self.pointset_size, self.output_dim))
        return torch.reshape(y, (self.pointset_size, self.output_dim))


    def initialize_residual(self):
        nn.init.zeros_(self.network[-1].linear.weight)
        nn.init.zeros_(self.network[-1].linear.bias)


