"""Utilities

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.utils as vutils


# Normalize MNIST dataset.
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def one_hot_encode(target, length):
    """Converts batches of class indices to classes of one-hot vectors."""
    batch_s = target.size(0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def checkpoint(state, epoch):
    """Save checkpoint"""
    model_out_path = 'results/trained_model/model_epoch_{}.pth'.format(epoch)
    torch.save(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))


def load_mnist(args):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cuda else {}

    print('===> Loading training datasets')
    training_set = datasets.MNIST(
        './data', train=True, download=True, transform=data_transform)
    training_data_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    print('===> Loading testing datasets')
    testing_set = datasets.MNIST(
        './data', train=False, download=True, transform=data_transform)
    testing_data_loader = DataLoader(
        testing_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return training_data_loader, testing_data_loader


def squash(sj, dim=2):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0

    This implement equation 1 from the paper.
    """
    sj_mag_sq = torch.sum(sj**2, dim, keepdim=True)
    # ||sj ||
    sj_mag = torch.sqrt(sj_mag_sq)
    v_j = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
    return v_j


def mask(out_digit_caps, cuda_enabled=True):
    """
    In the paper, they mask out all but the activity vector of the correct digit capsule.

    This means:
    a) during training, mask all but the capsule (1x16 vector) which match the ground-truth.
    b) during testing, mask all but the longest capsule (1x16 vector).

    :param out_digit_caps: tensor output of Digit Capsule layer
    """
    # a) Get capsule outputs lengths, ||v_c||
    v_length = torch.sqrt((out_digit_caps**2).sum(dim=2))

    # b) Pick out the index of longest capsule output, v_length by
    # masking the tensor by the max value in dim=1.
    _, max_index = v_length.max(dim=1)
    max_index = max_index.data

    # Method 1: masking with y.
    # c) In all batches, get the most active capsule
    # It's not easy to understand the indexing process with max_index
    # as we are 3D animal.
    batch_size = out_digit_caps.size(0)
    masked_v = [None] * batch_size
    for batch_ix in range(batch_size):
        # Batch sample
        sample = out_digit_caps[batch_ix]

        # Masks out the other capsules in this sample.
        v = Variable(torch.zeros(sample.size()))
        if cuda_enabled:
            v = v.cuda()

        # Get the maximum capsule index from this batch sample.
        max_caps_index = max_index[batch_ix]
        v[max_caps_index] = sample[max_caps_index]
        masked_v[batch_ix] = v # append v to masked_v

    # Concatenates sequence of masked capsules tensors along the batch dimension.
    masked = torch.stack(masked_v, dim=0)

    return masked


def save_image(image, file_name):
    """
    Save a given image into an image file
    """
    # Check number of channels in an image.
    if image.size(1) == 1:
        # Grayscale
        image_tensor = image.data.cpu() # get Tensor from Variable

    vutils.save_image(image_tensor, file_name)
