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
    - Tested with version 0.2.0.post4.
    - Code will not run with version 0.1.2 due to `keepdim` not available in this version.
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
| Log interval | 10 | --log-interval 10 |
| Disables CUDA training | false | --no-cuda |
| Num. of convolutional channel | 256 | --num-conv-channel 256 |
| Num. of primary unit | 8 | --num-primary-unit 8 |
| Primary unit size | 1152 | --primary-unit-size 1152 |
| Output unit size | 16 | --output-unit-size 16 |
| Num. routing iteration | 3 | --num-routing 3 |

## Results
Coming soon!

- training loss
```
# Log from the end of the last epoch.

... ... ... ... ... ... ... ... ... ... ...
... ... ... ... ... ... ... ... ... ... ...
Epoch: 10 [54912/60000 (91%)]   Loss: 0.039524
Epoch: 10 [55040/60000 (92%)]   Loss: 0.022957
Epoch: 10 [55168/60000 (92%)]   Loss: 0.039683
Epoch: 10 [55296/60000 (92%)]   Loss: 0.029625
Epoch: 10 [55424/60000 (92%)]   Loss: 0.038952
Epoch: 10 [55552/60000 (93%)]   Loss: 0.042668
Epoch: 10 [55680/60000 (93%)]   Loss: 0.048452
Epoch: 10 [55808/60000 (93%)]   Loss: 0.044467
Epoch: 10 [55936/60000 (93%)]   Loss: 0.023401
Epoch: 10 [56064/60000 (93%)]   Loss: 0.033448
Epoch: 10 [56192/60000 (94%)]   Loss: 0.033800
Epoch: 10 [56320/60000 (94%)]   Loss: 0.032488
Epoch: 10 [56448/60000 (94%)]   Loss: 0.027381
Epoch: 10 [56576/60000 (94%)]   Loss: 0.067512
Epoch: 10 [56704/60000 (94%)]   Loss: 0.044439
Epoch: 10 [56832/60000 (95%)]   Loss: 0.050315
Epoch: 10 [56960/60000 (95%)]   Loss: 0.044815
Epoch: 10 [57088/60000 (95%)]   Loss: 0.036546
Epoch: 10 [57216/60000 (95%)]   Loss: 0.030145
Epoch: 10 [57344/60000 (96%)]   Loss: 0.039890
Epoch: 10 [57472/60000 (96%)]   Loss: 0.049812
Epoch: 10 [57600/60000 (96%)]   Loss: 0.036459
Epoch: 10 [57728/60000 (96%)]   Loss: 0.045392
Epoch: 10 [57856/60000 (96%)]   Loss: 0.026301
Epoch: 10 [57984/60000 (97%)]   Loss: 0.037954
Epoch: 10 [58112/60000 (97%)]   Loss: 0.036555
Epoch: 10 [58240/60000 (97%)]   Loss: 0.035145
Epoch: 10 [58368/60000 (97%)]   Loss: 0.025339
Epoch: 10 [58496/60000 (97%)]   Loss: 0.037162
Epoch: 10 [58624/60000 (98%)]   Loss: 0.028909
Epoch: 10 [58752/60000 (98%)]   Loss: 0.034925
Epoch: 10 [58880/60000 (98%)]   Loss: 0.033485
Epoch: 10 [59008/60000 (98%)]   Loss: 0.018011
Epoch: 10 [59136/60000 (99%)]   Loss: 0.048944
Epoch: 10 [59264/60000 (99%)]   Loss: 0.022608
Epoch: 10 [59392/60000 (99%)]   Loss: 0.041117
Epoch: 10 [59520/60000 (99%)]   Loss: 0.046873
Epoch: 10 [59648/60000 (99%)]   Loss: 0.035419
Epoch: 10 [59776/60000 (100%)]  Loss: 0.029488
Epoch: 10 [44928/60000 (100%)]  Loss: 0.045561
```

- evaluation accuracy
```
Test set: Average loss: 0.0004, Accuracy: 9885/10000 (99%)
Checkpoint saved to model_epoch_10.pth
```

## TODO
- [DONE] Publish results.
- [WIP] More testing.
- Separate training and evaluation into independent command.
- Jupyter Notebook version.
- Create a sample to show how we can apply CapsNet to real-world application.
- Experiment with CapsNet:
    * Try using another dataset.
    * Come out a more creative model structure.
- Pre-trained model and weights.
- Add visualization for training and evaluation metrics.
- Implement recontruction loss.

## Credits

Referenced these implementations mainly for sanity check:
1. [TensorFlow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)
