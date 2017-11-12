# PyTorch CapsNet: Capsule Network for PyTorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/cedrickchee/capsule-net-pytorch/blob/master/LICENSE)
![completion](https://img.shields.io/badge/completion%20state-95%25-green.svg?style=plastic)

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

```bash
$ git clone https://github.com/cedrickchee/capsule-net-pytorch.git
$ cd capsule-net-pytorch
$ pip install -r requirements.txt
```

**Step 2.** 
Start the CapsNet on MNIST training and evaluation:

- Training with default settings:
```bash
$ python main.py
```

- Training on 8 GPUs with 30 epochs and 1 routing iteration:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --epochs 30 --num-routing 1 --threads 16 --batch-size 128 --test-batch-size 128
```

**Step 3.**
Test a pre-trained model:

If you have trained a model in Step 2 above, then the trained model will be saved to `results/trained_model/model_epoch_10.pth`. [WIP] Now just run the following command to get test results.

```bash
$ python main.py --is-training 0 --weights results/trained_model/model_epoch_10.pth
```

You can download the pre-trained model from my [Google Drive](https://drive.google.com/uc?export=download&id=1ojmG1nEkQKGPKO9lJr5gIupvNnKnyZ87).

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
| Num. of channels produced by the convolution | 256 | --num-conv-out-channel 256 |
| Num. of input channels to the convolution | 1 | --num-conv-in-channel 1 |
| Num. of primary unit | 8 | --num-primary-unit 8 |
| Primary unit size | 1152 | --primary-unit-size 1152 |
| Num. of digit classes | 10 | --num-classes 10 |
| Output unit size | 16 | --output-unit-size 16 |
| Num. routing iteration | 3 | --num-routing 3 |
| Regularization coefficient for reconstruction loss | 0.0005 | --regularization-scale 0.0005 |

## Results

### Test error

CapsNet classification test error on MNIST. The MNIST average and standard deviation results are reported from 3 trials.

[WIP] The results can be reproduced by running the following commands.

```bash
 python main.py --num-routing 1 --regularization-scale 0.0      #CapsNet-v1
 python main.py --num-routing 1 --regularization-scale 0.0005   #CapsNet-v2
 python main.py --num-routing 3 --regularization-scale 0.0      #CapsNet-v3
 python main.py --num-routing 3 --regularization-scale 0.0005   #CapsNet-v4
```

Method | Routing | Reconstruction | MNIST (%) | *Paper*
:---------|:------:|:---:|:----:|:----:
Baseline |  -- | -- | -- | *0.39*
CapsNet-v1 | 1 | no | -- | *0.34 (0.032)*
CapsNet-v2 | 1 | yes | -- | *0.29 (0.011)*
CapsNet-v3 | 3 | no | -- | *0.35 (0.036)*
CapsNet-v4 | 3 | yes | -- | *0.25 (0.005)*

### Training loss

Log from the end of the last epoch. View the full log [here](results/training_testing_log.txt).

```text
... ... ... ... ... ... ... ... ... ... ...
... ... ... ... ... ... ... ... ... ... ...
Epoch: 10 [0/60000 (0%)]        Loss: 0.235540
Epoch: 10 [1280/60000 (2%)]     Loss: 0.233107
Epoch: 10 [2560/60000 (4%)]     Loss: 0.232818
Epoch: 10 [3840/60000 (6%)]     Loss: 0.248812
Epoch: 10 [5120/60000 (9%)]     Loss: 0.246014
Epoch: 10 [6400/60000 (11%)]    Loss: 0.237645
Epoch: 10 [7680/60000 (13%)]    Loss: 0.248725
Epoch: 10 [8960/60000 (15%)]    Loss: 0.237840
Epoch: 10 [10240/60000 (17%)]   Loss: 0.246938
Epoch: 10 [11520/60000 (19%)]   Loss: 0.247348
Epoch: 10 [12800/60000 (21%)]   Loss: 0.246758
Epoch: 10 [14080/60000 (23%)]   Loss: 0.246680
Epoch: 10 [15360/60000 (26%)]   Loss: 0.253511
Epoch: 10 [16640/60000 (28%)]   Loss: 0.232439
Epoch: 10 [17920/60000 (30%)]   Loss: 0.229010
Epoch: 10 [19200/60000 (32%)]   Loss: 0.241444
Epoch: 10 [20480/60000 (34%)]   Loss: 0.239509
Epoch: 10 [21760/60000 (36%)]   Loss: 0.235857
Epoch: 10 [23040/60000 (38%)]   Loss: 0.240081
Epoch: 10 [24320/60000 (41%)]   Loss: 0.233029
Epoch: 10 [25600/60000 (43%)]   Loss: 0.239576
Epoch: 10 [26880/60000 (45%)]   Loss: 0.252535
Epoch: 10 [28160/60000 (47%)]   Loss: 0.243013
Epoch: 10 [29440/60000 (49%)]   Loss: 0.264241
Epoch: 10 [30720/60000 (51%)]   Loss: 0.241051
Epoch: 10 [32000/60000 (53%)]   Loss: 0.247486
Epoch: 10 [33280/60000 (55%)]   Loss: 0.238380
Epoch: 10 [34560/60000 (58%)]   Loss: 0.253946
Epoch: 10 [35840/60000 (60%)]   Loss: 0.258566
Epoch: 10 [37120/60000 (62%)]   Loss: 0.244170
Epoch: 10 [38400/60000 (64%)]   Loss: 0.240550
Epoch: 10 [39680/60000 (66%)]   Loss: 0.232219
Epoch: 10 [40960/60000 (68%)]   Loss: 0.233181
Epoch: 10 [42240/60000 (70%)]   Loss: 0.246600
Epoch: 10 [43520/60000 (72%)]   Loss: 0.235462
Epoch: 10 [44800/60000 (75%)]   Loss: 0.246548
Epoch: 10 [46080/60000 (77%)]   Loss: 0.234177
Epoch: 10 [47360/60000 (79%)]   Loss: 0.240156
Epoch: 10 [48640/60000 (81%)]   Loss: 0.246746
Epoch: 10 [49920/60000 (83%)]   Loss: 0.232246
Epoch: 10 [51200/60000 (85%)]   Loss: 0.237809
Epoch: 10 [52480/60000 (87%)]   Loss: 0.250668
Epoch: 10 [53760/60000 (90%)]   Loss: 0.233228
Epoch: 10 [55040/60000 (92%)]   Loss: 0.245191
Epoch: 10 [56320/60000 (94%)]   Loss: 0.251059
Epoch: 10 [57600/60000 (96%)]   Loss: 0.236024
Epoch: 10 [58880/60000 (98%)]   Loss: 0.236005
```

### Evaluation accuracy
```text
Test set: Average loss: 0.0020, Accuracy: 9908/10000 (99%)
Checkpoint saved to model_epoch_10.pth
```

### Reconstruction

The results of CapsNet-v4.

Digits at left are reconstructed images.
<table>
  <tr>
    <td>
     <img src="results/reconstructed_images.png"/>
    </td>
    <td>
    </td>
  </tr>
</table>

## TODO
- [DONE] Publish results.
- [DONE] More testing.
- Separate training and evaluation into independent command.
- Jupyter Notebook version.
- Create a sample to show how we can apply CapsNet to real-world application.
- Experiment with CapsNet:
    * Try using another dataset.
    * Come out a more creative model structure.
- Pre-trained model and weights.
- Add visualization for training and evaluation metrics.
- [DONE] Implement recontruction loss.

## Credits

Referenced these implementations mainly for sanity check:
1. [TensorFlow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)
