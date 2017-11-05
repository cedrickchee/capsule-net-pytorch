"""CapsNet Architecture

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from conv_layer import ConvLayer
from capsule_layer import CapsuleLayer


class Net(nn.Module):
    """
    A simple CapsNet with 3 layers
    """

    def __init__(self, num_conv_channel, num_primary_unit, primary_unit_size,
                 output_unit_size, num_routing, cuda_enabled):
        """
        In the constructor we instantiate one ConvLayer module and two CapsuleLayer modules
        and assign them as member variables.
        """
        super(Net, self).__init__()

        self.cuda_enabled = cuda_enabled

        self.conv1 = ConvLayer(in_channel=1,
                               out_channel=num_conv_channel,
                               kernel_size=9)

        # PrimaryCaps
        self.primary = CapsuleLayer(in_unit=0,
                                    in_channel=num_conv_channel,
                                    num_unit=num_primary_unit,
                                    unit_size=primary_unit_size,
                                    use_routing=False,
                                    num_routing=num_routing,
                                    cuda_enabled=cuda_enabled)

        # DigitCaps
        self.digits = CapsuleLayer(in_unit=num_primary_unit,
                                   in_channel=primary_unit_size,
                                   num_unit=10,
                                   unit_size=output_unit_size,
                                   use_routing=True,
                                   num_routing=num_routing,
                                   cuda_enabled=cuda_enabled)

    def forward(self, x):
        """
        Defines the computation performed at every forward pass.
        """
        out_conv1 = self.conv1(x)
        out_primary_caps = self.primary(out_conv1)
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, input, target, size_average=True):
        """Custom loss function"""
        m_loss = self.margin_loss(input, target, size_average)
        return m_loss

    def margin_loss(self, input, target, size_average=True):
        """Margin loss for digit existence
        """
        batch_size = input.size(0)

        # Implement equation 4 in the paper.

        # ||vc||
        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms.
        zero = Variable(torch.zeros(1))
        if self.cuda_enabled:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)
        t_c = target
        # Lc is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        if size_average:
            l_c = l_c.mean()

        return l_c
