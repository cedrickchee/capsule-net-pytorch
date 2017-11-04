# PyTorch CapsNet: Capsule Network for PyTorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/cedrickchee/capsule-net-pytorch/blob/master/LICENSE)
![completion](https://img.shields.io/badge/completion%20state-90%25-green.svg?style=plastic)

A PyTorch implementation of CapsNet (Capsule Network) based on this paper:
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)

Codes comes with ample comments and Python docstring.

**Status and Latest Updates:**

See the [CHANGELOG](CHANGELOG.md)

**Datasets**

The model was trained on the standard [MNIST](http://yann.lecun.com/exdb/mnist/) data.

*Note: you don't have to manually download and process the MNIST dataset as PyTorch will take care of this step for you.*

## Requirements
- Python
- [PyTorch](http://pytorch.org/)

## Usage

### Training and Evaluation
**Step 1.**
Clone this repository with ``git``.

```
$ git clone https://github.com/cedrickchee/capsule-net-pytorch.git
$ cd capsule-net-pytorch
```

**Step 2.** 
Start the training and evaluation:
```
$ python main.py
```

## Results
Coming soon!

- training loss
![total_loss](internal/img/training/training_loss.png)

![margin_loss](internal/img/training/margin_loss.png)
![reconstruction_loss](internal/img/training/reconstruction_loss.png)

- evaluation accuracy
![test_img1](internal/img/evaluation/test_000.png)

**TODO**
- [WIP] Publish results.
- [WIP] More testing.
- Separate training and evaluation into independent command.
- Jupyter Notebook version
- Create a sample to show how we can apply CapsNet to real-world application.
- Experiment with CapsNet:
    * Try using another dataset
    * Come out a more creative model structure
