## The shape and simplicity biases of adversarially robust ImageNet-trained CNN

This repository contains source code necessary to reproduce some of the main results in our paper:

Chen*, Agarwal*, Nguyen* (2020). _The shape and simplicity biases of adversarially robust ImageNet-trained CNNs_. [paper](https://arxiv.org/abs/2006.09373) | [code](https://github.com/anguyen8/shape-bias)

**If you use this software, please consider citing:**

    @article{chen2020shape,
      title={The shape and simplicity biases of adversarially robust ImageNet-trained CNNs},
      author={Chen, Peijie and Agarwal, Chirag and Nguyen, Anh},
      journal={arXiv preprint arXiv:2006.09373},
      year={2020}
    }

### 1. Robustness package
We use [Madry's robustness packages](https://github.com/MadryLab/robustness) to train the robust models for our experiment. This repo already include their packages (in robustness folder), so you don't need to download or install the robustness packages.

### 2. Pre-trained models
* The pretrained models are avaliable [here](https://drive.google.com/drive/u/0/folders/1KdJ0aK0rPjmowS8Swmzxf8hX6gU5gG2U).

### 3. Requirements
We recommend using Anaconda to create new enviroment:
```
conda env create -f enviroment.yml -n {NAME OF NEW ENV}
```

### 4. Network Dissection
Our Network Dissection experiment was developed on top of Bolei Zhou's [network dissection](https://github.com/CSAILVision/NetDissect-Lite).

* In order to run Network Dissection, you will need follow [network dissection](https://github.com/CSAILVision/NetDissect-Lite) repo to download the Broden dataset using:
```
    ./script/dlbroden.sh
    ./script/dlzoo_example.sh
```
* Setup the model, target layers and other settings in [settings/settings.py](settings/settings.py).
* Then run:
```
    python netdissect.py
```

### MIT License
Code in this repository is licensed under MIT License.
