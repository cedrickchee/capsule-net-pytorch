"""Capsule layer

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils


class CapsuleLayer(nn.Module):
    """
    The core implementation of the idea of capsules
    """

    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled):
        super(CapsuleLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled

        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and DigitCaps).
            No routing is used between Conv1 and PrimaryCapsules.

            This means PrimaryCapsules is composed of several convolutional units.
            """
            # self.conv_units = [self.conv_unit(u) for u in range(self.num_unit)]
            self.conv_units = nn.ModuleList([
                nn.Conv2d(self.in_channel, 32, 9, 2) for u in range(self.num_unit)
            ])

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def routing(self, x):
        """
        Routing algorithm for capsule.

        :return: vector output of capsule j
        """
        batch_size = x.size(0)

        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
        weight = torch.cat([self.weight] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        u_hat = torch.matmul(weight, x)

        # All the routing logits (b_ij in the paper) are initialized to zero.
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
        if self.cuda_enabled:
            b_ij = b_ij.cuda()

        # From the paper in the "Capsules on MNIST" section,
        # the sample MNIST test reconstructions of a CapsNet with 3 routing iterations.
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            # Routing algorithm

            # Calculate routing or also known as coupling coefficients (c_ij).
            c_ij = F.softmax(b_ij)  # Convert routing logits (b_ij) to softmax.
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Implement equation 2 in the paper.
            # u_hat is weighted inputs
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = utils.squash(s_j)

            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)

    def no_routing(self, x):
        """
        Get output for each unit.
        A unit has batch, channels, height, width.

        :return: vector output of capsule j
        """
        # unit = [self.conv_units[i](x) for i in range(self.num_unit)]
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]

        # Stack all unit outputs.
        unit = torch.stack(unit, dim=1)

        # Flatten
        unit = unit.view(x.size(0), self.num_unit, -1)

        # Return squashed outputs.
        return utils.squash(unit)

    def conv_unit(self, idx):
        """
        Create a convolutional unit.

        A convolutional unit uses normal convolutional layer with a nonlinearity (squash).
        """
        unit = nn.Conv2d(in_channels=self.in_channel,
                         out_channels=32,
                         kernel_size=9,
                         stride=2)
        self.add_module("conv_unit" + str(idx), unit)
        return unit
