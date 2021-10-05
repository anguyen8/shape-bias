from settings import alblation_setting as alb_setting, eval_setting as settings
import torch
from robustness.datasets import DATASETS
# from robustness.model_utils import make_and_restore_model, make_and_restore_model_custom
from robustness.model_utils import make_and_restore_model
import torchvision
import pandas as pd
import os
import torchvision.transforms as transforms
from PIL import Image
from loader.models import alexnet, resnet50
import torch.nn as nn
import ipdb

def loadmodel(model_name = settings.MODEL, model_path = settings.MODEL_PATH,
              Madry_model = settings.MADRYMODEL, model_parallel=settings.MODEL_PARALLEL, GPU= True,
              num_of_classes = settings.NUM_CLASSES):
    if Madry_model:
        '''load Madry model'''
        dataset = DATASETS['imagenet']('robustness/dataset')
        
        model, checkpoint = make_and_restore_model(arch=model_name[:-2] if model_name[-1] in ['r', 'R'] else model_name,
                                                   dataset=dataset, parallel=model_parallel,
                                                   resume_path=model_path)
        # try to handle Madry model as regular model
        # if model_name == 'googlenet-r':
        #     model = model.module.model
        # elif model_name == 'resnet50-r':
        #     model = model.module.model
        # elif model_name == 'alexnet-r':
        #     model = model.module.model
        # model = model.module.model
    else: # load regular model from pytorch
        checkpoint = torch.load(model_path)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            if str.lower(model_name) == 'googlenet':
                model = torchvision.models.__dict__[model_name](num_classes=num_of_classes,
                                                                    aux_logits=settings.AUXILIARY)
            else:
                model = torchvision.models.__dict__[model_name](num_classes=num_of_classes)
            if model_parallel:
                if 'stat_dict' in checkpoint.keys():
                    checkpoint = checkpoint['stat_dict']
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    if GPU:
        # model = nn.DataParallel(model)
        model.cuda()
    model.eval()
    return model

def alblation_load_model(model_name =alb_setting.MODEL, model_path=alb_setting.MODEL_PATH,
                         Madry_model=alb_setting.MADRYMODEL, model_parallel=alb_setting.MODEL_PARALLEL, GPU=True,
                         zero_dict=None, keep=False):
    if Madry_model:
        '''load Madry model'''
        dataset = DATASETS['imagenet']('robustness/dataset')
        model, checkpoint = make_and_restore_model_custom(arch=model_name[:-2],
                                                   dataset=dataset, parallel=model_parallel,
                                                   resume_path=model_path, zero_dict=zero_dict, keep=keep)
        # try to handle Madry model as regular model
        if model_name == 'googlenet-r':
            model = model.model
        elif model_name == 'resnet50-r':
            model = model.module.model
        elif model_name == 'alexnet-r':
            model = model.module.model
    else: # load regular model
        checkpoint = torch.load(model_path)
        if model_name == 'alexnet':
            model = alexnet.AlexNet(zero_dict=zero_dict, keep=keep)
        elif model_name == 'resnet50':
            model = resnet50.resnet50(zero_dict=zero_dict, keep=keep)
        # elif model_name == 'googlenet':
        #     model = googlenet.googlenet(zero_dict=zero_dict, keep=keep)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            if model_parallel:
                if 'stat_dict' in checkpoint.keys():
                    checkpoint = checkpoint['stat_dict']
                state_dict = {str.replace(k, 'module.', ''): v for k, v in
                              checkpoint.items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
    if GPU:
        model.cuda()
        # model = nn.DataParallel(model)
    # model.eval()
    return model