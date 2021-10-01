import torchvision.datasets as datasets
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import os, shutil, argparse
import ipdb
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils import check_path, scrimble_img
from settings import settings
from loader.model_loader import loadmodel, loadrobust
from loader.class_loader import accuracy
from plotHelper import layerNames
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS

channel_activation = []
def hook_feature(module, input, output):
    channel_activation.append(output.data.cpu().numpy())

def image_name_to_netid():
    # Map imagenet names to their netids
    input_f = open("ILSVRC2012/imagenet_validation_imagename_labels.txt")
    label_map = {}
    netid_map = {}
    for line in input_f:
        parts = line.strip().split(" ")
        label_map[parts[0]] = parts[1]
        netid_map[parts[0]] = parts[2]
    return label_map, netid_map

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)




def find_activation(data_path, out_path, model_name, model_path, layers = None, Madry_model=False, data_type=None, resize=True, normalizing=True):
    if Madry_model:
        dataset = DATASETS['imagenet']('robustness/dataset')
        model, checkpoint = make_and_restore_model(arch=model_name[:-2],
                                                   dataset=dataset, parallel=settings.MODEL_PARALLEL,
                                                   resume_path=model_path)

        model = loadrobust(hook_feature, model, checkpoint, layers, model_name, parallel=True)
    else:
        model = loadmodel(hook_feature, model_name, model_path, layers)

    batch_size = 10
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = []
    if resize:
        trans.append(transforms.Resize(256))
        trans.append(transforms.CenterCrop(224))
    trans.append(transforms.ToTensor())
    if normalizing:
        trans.append(normalize)


    dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path, transforms.Compose(trans)),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)



    with torch.no_grad():


        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # evaluate images
        # pbar = tqdm(total=len(dataloader))
        # pbar.set_description(f"Classifying {model_name}")
        for i, (img, target) in enumerate(dataloader):
            if i > 1:
                break
            target_tensor = torch.tensor(list(target)).to('cuda')

            img = img.to('cuda')
            score = model(img)
            # Compute confidence
            if model_name[-1] in ['r', 'R']:
                score = score[0]
            prob = torch.nn.functional.softmax(score, dim=1)
            (acc1, acc5), correct_top1 = accuracy(prob, target_tensor, (1, 5))
            imgs = dataloader.dataset.imgs[:batch_size]
            top1.update(acc1[0])
            top5.update(acc5[0])

            # Display progress
        #     pbar.update(1)
        # pbar.close()

    # save feature output
    # save path
    activation_path = os.path.join(out_path, model_name)
    check_path(activation_path)
    file_name = f"{activation_path}/{args.data_type}_activations.npy"
    layer_list = layerNames(model_name)
    activation_dict = {}
    # data break down due to data parallel
    restore_data = []
    i = 0
    if model_name[-1] in ['r', 'R']:
        for idx, data in enumerate(channel_activation):
            if idx < 25:
                if idx >= 5:
                    restore_data[idx%5] = np.vstack((restore_data[idx%5], data))
                else:
                    restore_data.append(data)
            #     if idx >= 5:
            #         restore_data[idx%5] = np.vstack((restore_data[idx%5], data[np.newaxis, 1]))
            #     else:
            #         restore_data.append(data[np.newaxis, 1])
            # else:
            #     if restore_data[i].shape[0] == 10:
            #         i += 1
            #     restore_data[i] = np.vstack((restore_data[i], data[np.newaxis, 1]))
        activation_data = restore_data
    else:
        activation_data = channel_activation

    # save channel excitation
    for idx, layer_name in enumerate(layers):
        activation_dict[layer_list[layer_name]] = activation_data[idx]
    np.save(file_name, activation_dict)
    # print(f'{model_name}: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%')
    return imgs


def plotActivation(file_path, out_path):
    activations = np.load(file_path, allow_pickle=True).item()
    font = ImageFont.truetype("arial.ttf", 10)
    # plot for each layer
    for key, activation in activations.items():
        print(f'Plotting {key}')
        number_of_imgs, number_of_channels, img_shape_x, img_shape_y = activation.shape
        # plot for each image
        for img_number in tqdm(range(number_of_imgs)):
            # plot for each channel
            for channel_number in range(number_of_channels):
                channel_plot = Image.new('L', (img_shape_x, img_shape_y), 255)
                channel_output = activation[img_number, channel_number]
                # pixel = ((channel_output - channel_output.min())/np.ptp(channel_output)).astype(np.uint8)*255
                if channel_output.max() != 0:
                    pixel = (channel_output*(255/(channel_output.max()))).astype(np.uint8)
                else:
                    pixel = (channel_output).astype(np.uint8)*255
                img = Image.fromarray(pixel, mode='L')
                channel_plot.paste(img, (0, 0))
                # massage = f'{channel_number:04d}'
                # draw = ImageDraw.Draw(channel_plot)
                # draw.text((0, img_shape_y-2), massage, font = font)
                channel_plot.save(f'{out_path}/img{img_number:02d}_{key}_{channel_number:04d}.jpg')

def argParser():
    parser = argparse.ArgumentParser(description='Save channel activation and plot')
    parser.add_argument('--network', default='alexnet',  help='Model name')
    parser.add_argument('--model_path', default='zoo/alexnet.pth', help='Model path')
    parser.add_argument('--data_path', default='dataset/correct/alexnet_clean_super_inter', help='path to dataset')
    parser.add_argument('--madry', default=False, type=bool, help='weather it is a madry model')
    parser.add_argument('--data_type', default='clean', help='Model name')
    parser.add_argument('--resize', default=True, type=bool, help='resize or not')
    parser.add_argument('--normalize', default=True, type=bool, help='normalize or not')
    parser.add_argument('--out_path', default='result/channel_activation', help='output path')
    parser.add_argument('--layers', default=['1', '4', '7', '9', '11'], help='Target layers')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argParser()
    # img_path = '/home/chirag/convergent_learning/data/val/'
    # img_path = 'dataset/gaussian_noise'
    # img_path = '/home/chirag/stylized_imagenet/val'
    # img_path = '/home/chirag/convergent_learning/scrambling_dataset_112'

    imgs = find_activation(args.data_path, args.out_path, args.network, args.model_path, layers=args.layers, Madry_model=args.madry,
                     data_type=args.data_type, resize=args.resize, normalizing=args.normalize)


    # plot channel activations
    img_path = f'result/channel_activation/{args.network}/{args.data_type}'
    check_path(img_path)
    plotActivation(f'result/channel_activation/{args.network}/{args.data_type}_activations.npy',  img_path)

    # copy org images
    org_img_path = f'{args.out_path}/{args.data_type}_img'
    check_path(org_img_path)
    if len(os.listdir(org_img_path)) != 10:
        for i, img in enumerate(imgs):
            shutil.copy(f'{img[0]}', f'{org_img_path}/img{i:02d}.jpg')


    exit(0)