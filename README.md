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

### 5. A toy sample of loading and hook features from robust model. 
* This sample is avaliable at [load_sample.py](load_sample.py): 

```python
from settings import settings
from loader.model_loader import loadrobust
# for loading robust version
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS
import torch
from PIL import Image
from torchvision import transforms

# target layers can be changed in settings file. For this sample, it will hook layer ["1", "4", "7", "9", "11"] of AlexNet-R
features_blobs = []
def hook_feature(module, input, output): 
    features_blobs.append(output.data.cpu().numpy())

def sample_loading_and_hooking():
    global features_blobs
    # load robust model
    if settings.MODEL[-1] in ['r', 'R']:
        '''loading robust model'''
        dataset = DATASETS['imagenet']('robustness/dataset')
        model, checkpoint = make_and_restore_model(arch=settings.MODEL[:-2],
                                                dataset=dataset, parallel=settings.MODEL_PARALLEL,
                                                resume_path=settings.MODEL_FILE)
        # add hooker on target layers
        model = loadrobust(hook_feature, model, checkpoint,  settings.FEATURE_NAMES)
    
    img = Image.open('figures/real/ILSVRC2012_val_00002691.JPEG')

    TEST_TRANSFORMS_IMAGENET = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    input_tensor = TEST_TRANSFORMS_IMAGENET(img).unsqueeze(0).cuda()
    # make predictions
    logit, _ = model(input_tensor)
    pred_class = logit.argmax()
    # put hooked features in to a dict.
    hooked_features = dict(zip(["1", "4", "7", "9", "11"], features_blobs)) # dict for features of layer1, 4, 7, 9, 11.
    return logit, pred_class, hooked_features

if __name__ == '__main__':
    #* A naive sample for load and hook features from robust model
    logit, pred_class, hooked_features = sample_loading_and_hooking()
```
### 6. ImageNet-CL
We provide the index of [ImageNet-CL(.npy)](data/), where the robust model and standard model both make correct predictions. There are three .npy files, one for each pair of network. i.e. (AlexNet, AlexNet-R). 

* Sample:
```python
import numpy as np
alexnet_intersect_img_idx = np.load('data/intersection_alexnet_alexnet-r.npy')
```
The .npy file contain a list of image names of ImageNet val set.
```
>>> print(intersect_img_idx)
['ILSVRC2012_val_00000003.JPEG' 'ILSVRC2012_val_00000007.JPEG'
 'ILSVRC2012_val_00000012.JPEG' ... 'ILSVRC2012_val_00049990.JPEG'
 'ILSVRC2012_val_00049991.JPEG' 'ILSVRC2012_val_00049999.JPEG']
```

<<<<<<< HEAD
### 7. Simple sample to evaluate robust model on ImageNet
To reproduce Table. 1 prediction accuracy of robust model in our paper, use the following script in genereal cases:
```
python evaluate_on_imaget.py --model_name {MODEL ARCHITECTURE} --model_path {CHECK POINT PATH} --data_path {PATH TO ILSVRC2012} --madry_model
```
for example,
```
python evaluate_on_imaget.py --model_name alexnet --model_path zoo/alexnet_r.pt --data_path path/to/ILSVRC2012 --madry_model
```
=======
### MIT License
2c4466fc3d4ebaf42b3a5ffa8a078685287db67b
