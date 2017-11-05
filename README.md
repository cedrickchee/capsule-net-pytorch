# PyTorch CapsNet: Capsule Network for PyTorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/cedrickchee/capsule-net-pytorch/blob/master/LICENSE)
![completion](https://img.shields.io/badge/completion%20state-90%25-green.svg?style=plastic)

A CUDA-enabled PyTorch implementation of CapsNet (Capsule Network) based on this paper:
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)

**What is a Capsule**

> A Capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part.

Codes comes with ample comments and Python docstring.

**Status and Latest Updates:**

See the [CHANGELOG](CHANGELOG.md)

**Datasets**

The model was trained on the standard [MNIST](http://yann.lecun.com/exdb/mnist/) data.

*Note: you don't have to manually download, preprocess, and load the MNIST dataset as [TorchVision](https://github.com/pytorch/vision) will take care of this step for you.*

## Requirements
- Python 3
- [PyTorch](http://pytorch.org/)
- TorchVision

## Usage

### Training and Evaluation
**Step 1.**
Clone this repository with ``git`` and install project dependencies.

```
$ git clone https://github.com/cedrickchee/capsule-net-pytorch.git
$ cd capsule-net-pytorch
$ pip install -r requirements.txt
```

**Step 2.** 
Start the training and evaluation:
```
$ python main.py
```

**The default hyper parameters:**

| Parameter | Value | CLI arguments |
| --- | --- | --- |
| Training epochs | 10 | --epochs 10 |
| Learning rate | 0.01 | --lr 0.01 |
| Training batch size | 128 | --batch-size 128 |
| Testing batch size | 128 | --test-batch-size 128 |
| Loss threshold | 0.001 | --loss-threshold 0.001 |
| Log interval | 1 | --log-interval 1 |
| Num. of convolutional channel | 256 | --num-conv-channel 256 |
| Num. of primary unit | 8 | --num-primary-unit 8 |
| Primary unit size | 1152 | --primary-unit-size 1152 |
| Output unit size | 16 | --output-unit-size 16 |

## Results
Coming soon!

- training loss
![total_loss](internal/img/training/training_loss.png)

![margin_loss](internal/img/training/margin_loss.png)
![reconstruction_loss](internal/img/training/reconstruction_loss.png)

- evaluation accuracy
![test_img1](internal/img/evaluation/test_000.png)

## TODO
- [WIP] Publish results.
- [WIP] More testing.
- Separate training and evaluation into independent command.
- Jupyter Notebook version.
- Create a sample to show how we can apply CapsNet to real-world application.
- Experiment with CapsNet:
    * Try using another dataset.
    * Come out a more creative model structure.
- Pre-trained model and weights.

## Credits

Referenced these implementations mainly for sanity check:
1. [TensorFlow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)
