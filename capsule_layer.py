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


class CapsuleLayer(nn.Module):
    """
    The core implementation of the idea of capsules
    """
    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing, cuda):
        super(CapsuleLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.cuda = cuda

        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            self.W = nn.Parameter(torch.randn(
                1, in_channel, num_unit, unit_size, in_unit))
        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and DigitCaps).
            No routing is used between Conv1 and PrimaryCapsules.

            This means PrimaryCapsules is composed of several convolutional units.
            So, implementation-wise, it uses normal convolutional layer with a nonlinearity (squash).
            """
            def create_conv_unit(idx):
                unit = nn.Conv2d(in_channels=in_channel,
                                 out_channels=32,
                                 kernel_size=9,
                                 stride=2)
                self.add_module("conv_unit" + str(idx), unit)
                return unit

            self.conv_units = [create_conv_unit(u) for u in range(self.num_unit)]

    @staticmethod
    def squash(sj):
        """
        Non-linear 'squashing' function.
        This implement equation 1 from the paper.
        """
        sj_mag_sq = torch.sum(sj**2, dim=2, keepdim=True)
        # ||sj ||
        sj_mag = torch.sqrt(sj_mag_sq)
        v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
        return v_j

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
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        u_hat = torch.matmul(W, x)

        # All the routing logits (b_ij in the paper) are initialized to zero.
        b_ij = Variable(torch.zeros(
            1, self.in_channel, self.num_unit, 1))
        if self.cuda:
            b_ij = b_ij.cuda()

        # From the paper in the "Capsules on MNIST" section,
        # the sample MNIST test reconstructions of a CapsNet with 3 routing iterations.
        num_iterations = 3

        for iteration in range(num_iterations):
            # Routing algorithm

            # Calculate routing or also known as coupling coefficients (c_ij).
            c_ij = F.softmax(b_ij)  # Convert routing logits (b_ij) to softmax.
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Implement equation 2 in the paper.
            # u_hat is weighted inputs
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = CapsuleLayer.squash(s_j)

            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(
                4).mean(dim=0, keepdim=True)

            # Update routing (b_ij)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)

    def no_routing(self, x):
        """
        Get output for each unit.
        A unit has batch, channels, height, width.

        :return: vector output of capsule j
        """
        unit = [self.conv_units[i](x) for i in range(self.num_unit)]

        # Stack all unit outputs.
        unit = torch.stack(unit, dim=1)

        # Flatten
        unit = unit.view(x.size(0), self.num_unit, -1)

        # Return squashed outputs.
        return CapsuleLayer.squash(unit)
