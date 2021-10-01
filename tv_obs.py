# dependencies
import re
import os
import ipdb
import argparse
import torch
import collections
from tqdm import tqdm
# import cv2
import time
import pickle
import numpy as np
import torch.utils.data
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from torch.utils import model_zoo
import torchvision
from matplotlib import cm

# input arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet validation')
parser.add_argument('--model', default='alexnet',
                    help='model name')
parser.add_argument('--ty', type=str, required=True, help='clean | noisy')


def load_pickle(x):
    return np.load(x, allow_pickle=True)


def tv_norm(x, tv_beta=1):
    img = torch.tensor(x)
    row_grad = torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta).sum()
    col_grad = torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta).sum()
    return row_grad + col_grad


def calculate_tv(x, path, key, noisy=False, robust=False):
    # newd = dict.fromkeys(x)

    if robust:
        if not noisy:
            key = f'module.module.model.{key}'
        else:
            key = f'module.model.{key}'
    else:
        if not noisy:
            key = f'module.{key}'

    tv_stats = []
    for f in x:
        # ipdb.set_trace()
        load_file = load_pickle(f'{path}/{f}')[key][0].squeeze()
        temp_tv = []
        for i in range(load_file.shape[0]):
            temp_tv.append(tv_norm(load_file[i, :]).item())
        tv_stats.append(temp_tv)
    tv_stats = np.array(tv_stats)
    return np.mean(tv_stats, axis=0)


def save_pickle(x, name):
    # Save file for dictionaries
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()


# function for setting the colors of the box plots pairs
from pylab import setp


def setBoxColors(bp):
    c1 = [52 / 255, 210 / 255, 235 / 255, 1]
    c2 = [145 / 255, 1 / 255, 1 / 255, 1]
    setp(bp['boxes'][0], color=c1)
    setp(bp['caps'][0], color=c1)
    setp(bp['caps'][1], color=c1)
    setp(bp['whiskers'][0], color=c1)
    setp(bp['whiskers'][1], color=c1)
    setp(bp['fliers'][0], color=c1)
    setp(bp['fliers'][1], color=c1)
    setp(bp['medians'][0], color='black')

    setp(bp['boxes'][1], color=c2)
    setp(bp['caps'][2], color=c2)
    setp(bp['caps'][3], color=c2)
    setp(bp['whiskers'][2], color=c2)
    setp(bp['whiskers'][3], color=c2)
    # setp(bp['fliers'][2], color='red')
    # setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='black')

# find all outliers
def findOutlier():
    s_dict = {}
    r_dict = {}
    for idx in range(5):
        tv_channel_alexnet_standard_clean = load_pickle(f'tv_stats/tv_channel_alexnet_standard_clean_{idx:02d}.pkl')
        tv_channel_alexnet_standard_noisy = load_pickle(f'tv_stats/tv_channel_alexnet_standard_noisy_{idx:02d}.pkl')
        tv_channel_alexnet_robust_clean = load_pickle(f'tv_stats/tv_channel_alexnet_robust_clean_{idx:02d}.pkl')
        tv_channel_alexnet_robust_noisy = load_pickle(f'tv_stats/tv_channel_alexnet_robust_noisy_{idx:02d}.pkl')
        key = [key for key in tv_channel_alexnet_standard_clean.keys()][idx]

        # find outliers
        threshold = tv_channel_alexnet_standard_clean[key].mean() * 0.2
        s_outliers = []
        s_tv_diff = []
        r_outliers = []
        r_tv_diff = []
        for channel_id in range(len(tv_channel_alexnet_standard_clean[key])):
            x, y = tv_channel_alexnet_standard_clean[key][channel_id], tv_channel_alexnet_standard_noisy[key][channel_id]
            if abs(x - y) >= threshold:
                s_outliers.append(channel_id)
                s_tv_diff.append(int(abs(x - y)))
            x, y = tv_channel_alexnet_robust_clean[key][channel_id], tv_channel_alexnet_robust_noisy[key][channel_id]
            if abs(x - y) >= threshold:
                r_outliers.append(channel_id)
                r_tv_diff.append(int(abs(x - y)))

        # sort by tv diff
        s_tv_stats = pd.DataFrame(np.array([s_outliers, s_tv_diff]).T, columns=['Channel_ID', 'TV_difference'])
        s_tv_stats.sort_values(by='TV_difference', ascending=False, inplace=True)
        s_tv_stats.reset_index(drop=True, inplace=True)
        r_tv_stats = pd.DataFrame(np.array([r_outliers, r_tv_diff]).T, columns=['Channel_ID', 'TV_difference'])
        r_tv_stats.sort_values(by='TV_difference', ascending=False, inplace=True)
        r_tv_stats.reset_index(drop=True, inplace=True)

        s_dict[f'conv{idx+1}'] = s_tv_stats.Channel_ID.to_list()
        r_dict[f'conv{idx+1}'] = r_tv_stats.Channel_ID.to_list()

    np.save(f'result/tv/alexnet_outliers.npy', s_dict)
    np.save(f'result/tv/alexnet_r_outliers.npy', r_dict)


def main():
    # parse input arguments
    args = parser.parse_args()

    data_to_plot_clean = []
    data_to_plot_noisy = []

    # Plot TV channels
    for idx in range(5):
        tv_channel_alexnet_standard_clean = load_pickle(f'tv_stats/tv_channel_alexnet_standard_clean_{idx:02d}.pkl')
        tv_channel_alexnet_standard_noisy = load_pickle(f'tv_stats/tv_channel_alexnet_standard_noisy_{idx:02d}.pkl')
        tv_channel_alexnet_robust_clean = load_pickle(f'tv_stats/tv_channel_alexnet_robust_clean_{idx:02d}.pkl')
        tv_channel_alexnet_robust_noisy = load_pickle(f'tv_stats/tv_channel_alexnet_robust_noisy_{idx:02d}.pkl')
        key = [key for key in tv_channel_alexnet_standard_clean.keys()][idx]
        if args.ty == 'noisy':
            mycm = cm.get_cmap('Set3', 2)
            plt.figure()
            plt.rcParams.update({'font.size': 15})  # , 'font.weight':'bold'})
            plt.rc("font", family="sans-serif")
            plt.scatter(tv_channel_alexnet_standard_clean[key], tv_channel_alexnet_standard_noisy[key], s=64,
                        label=f'AlexNet conv{idx + 1}', color=mycm.colors[0], edgecolors='k')
            plt.scatter(tv_channel_alexnet_robust_clean[key], tv_channel_alexnet_robust_noisy[key], s=64,
                        label=f'AlexNet-R conv{idx + 1}', color=mycm.colors[1], edgecolors='k')
            # annotate points
            threshold =  tv_channel_alexnet_standard_clean[key].mean()*0.2
            for channel_id in range(len(tv_channel_alexnet_standard_clean[key])):
                x, y = tv_channel_alexnet_standard_clean[key][channel_id], tv_channel_alexnet_standard_noisy[key][channel_id]
                if abs(x-y) >= threshold:
                    plt.annotate(channel_id, (x, y), fontsize=7)
                x, y = tv_channel_alexnet_robust_clean[key][channel_id], tv_channel_alexnet_robust_noisy[key][channel_id]
                if abs(x-y) >= threshold:
                    plt.annotate(channel_id, (x, y), fontsize=7)

            plt.xlabel('TV of channels for Clean Images')
            plt.ylabel('TV of channels for Noisy Images')
            x1 = np.max([np.max(tv_channel_alexnet_robust_clean[key]), np.max(tv_channel_alexnet_standard_clean[key])])
            x2 = np.max([np.max(tv_channel_alexnet_robust_noisy[key]), np.max(tv_channel_alexnet_standard_noisy[key])])
            plt.plot(np.arange(int(np.max([x1, x2]))), np.arange(int(np.max([x1, x2]))), '-k')
            plt.legend()
            out_dir = 'result/tv'
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            plt.savefig(f'{out_dir}/tv_{idx:02d}.pdf', bbox_inches='tight', pad=0)
        elif args.ty == 'clean':
            ## combine these different collections into a list
            data_to_plot_clean.append(
                [(tv_channel_alexnet_standard_clean[key][i] - tv_channel_alexnet_robust_clean[key][i]) for i in
                 range(len(tv_channel_alexnet_robust_clean[key]))])
            data_to_plot_noisy.append(
                [(tv_channel_alexnet_standard_noisy[key][i] - tv_channel_alexnet_robust_noisy[key][i]) for i in
                 range(len(tv_channel_alexnet_robust_noisy[key]))])

        # ipdb.set_trace()
        print(f'Done ch: {idx:02d}')

if __name__ == '__main__':
    # main()
    findOutlier()
    exit(0)