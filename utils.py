"""Utilities

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def normalize_dataset():
    """Normalize MNIST dataset."""
    return transforms.Compose([
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
    model_out_path = 'model_epoch_{}.pth'.format(epoch)
    torch.save(state, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))


def load_mnist(args):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    print('===> Loading training datasets')
    training_set = datasets.MNIST(
        '../data', train=True, download=True, transform=normalize_dataset)
    training_data_loader = DataLoader(
        training_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

    print('===> Loading testing datasets')
    testing_set = datasets.MNIST(
        '../data', train=False, download=True, transform=normalize_dataset)
    testing_data_loader = DataLoader(
        testing_set, num_workers=args.threads, batch_size=args.test_batch_size, shuffle=True)

    return training_data_loader, testing_data_loader
