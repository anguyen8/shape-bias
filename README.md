## The shape and simplicity biases of adversarially robust ImageNet-trained CNN

### Robust models
We use [Madry's robustness packages](https://github.com/MadryLab/robustness) to train the robust models for our experiment. This repo already include their packages (in robustness folder), so you don't need to download or install the robustness packages.

### Network dissection
Our network dissection experiment develop on top of Bolei Zhou's [network dissection](https://github.com/CSAILVision/NetDissect-Lite) work.

### Usage
* The pretrained models are avaliable [here](https://drive.google.com/drive/u/0/folders/1KdJ0aK0rPjmowS8Swmzxf8hX6gU5gG2U).
* In order to run network dissection, you will need follow [network dissection](https://github.com/CSAILVision/NetDissect-Lite) repo to download the Broden dataset using:
```
    ./script/dlbroden.sh
    ./script/dlzoo_example.sh
```
* setup the model, target layers and other settings in [settings/settings.py](settings/settings.py).
* Then run:
```
    python netdissect.py
```
### Requirements
We recommand use conda to create new enviroment:
```
conda env create -f enviroment.yml -n {NAME OF NEW ENV}
```
