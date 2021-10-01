import torch, torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from robustness.datasets import CIFAR, ImageNet
import torchvision.transforms as transforms
from robustness import model_utils
import ipdb
import sys
import numpy as np
import time
import os
from grad_utils import plot_cifar_grad
from utils import check_path, plot_grad
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import argparse


use_cuda = torch.cuda.is_available()
## For reproducebility
# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for meaningful perturbation explanation of the image')

    parser.add_argument('-idp', '--img_dir_path', default='data/cifar-10-batches-py',
                        help='Path to the input image dir', metavar='DIR')

    parser.add_argument('-op', '--out_path',
                        help='Path of the output directory where you want to save the results (Default is ./img_name/)')


    parser.add_argument('-ifp', '--if_pre', type=int, choices=range(2),
                        help='Before or after softmax', default=0,
                        )

    parser.add_argument('-n_seed', '--noise_seed', type=int,
                        help='Seed for the Gaussian noise. Default: 0', default=0,
                        )


    parser.add_argument('-bs', '--batch_size', type=int,
                        help='Size for the batch of images. Default: 1', default=1,
                        )

    parser.add_argument('--network', default='resnet18', help='Network name')

    parser.add_argument('--dataset', default='cifar10', help='Dataset')

    parser.add_argument('--pytorch_model', default=False, type=bool, help='If use pytorch pre-train model or not')

    parser.add_argument('--model_path', required=True, help='model stat dict')
    # Parse the arguments
    args = parser.parse_args()

    if args.noise_seed is not None:
        print(f'Setting the numpy seed with value: {args.noise_seed}')
        np.random.seed(args.noise_seed)

    if args.img_dir_path is None:
        print('Please provide path to image dir. Exiting')
        sys.exit(1)
    else:
        args.img_dir_path = os.path.abspath(args.img_dir_path)

    if args.out_path is None:
        args.out_path = './'
    args.out_path = os.path.abspath(args.out_path)

    args.batch_size = 1  ## to make sure only 1 image is being ran. you can chnage it if you like

    return args

def save_grad(network, model_path, dataset, out_path, save_org=False, org_path='result/grad_data/org', batch_size=1,
              pytorch_pretrained=False):
    if save_org:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transforms.Compose([transforms.ToTensor()]))
    if pytorch_pretrained:
        model_path = None
    model, checkpoint = model_utils.make_and_restore_model(arch=network,
                                                           dataset=dataset, parallel=True,
                                                           resume_path=model_path, pytorch_pretrained=pytorch_pretrained)
    if pytorch_pretrained:
        model.cuda()
    _, val_loader = dataset.make_loaders(batch_size=batch_size, workers=8, only_val=True,
                                                                      shuffle_val=False)
    # evaluate model
    pbar = tqdm(total=len(val_loader))
    pbar.set_description(f"saving image for {network}")
    for i, (img, targ_class) in enumerate(val_loader):
        if i >= 10:
            break
        model.zero_grad()
        targ_class = targ_class.cpu()

        if use_cuda:
            img = img.cuda()

        ## #We want to compute gradients
        img = Variable(img, requires_grad=True)

        ## #Prob and gradients
        sel_nodes_shape = targ_class.shape
        ones = torch.ones(sel_nodes_shape)
        if use_cuda:
            ones = ones.cuda()

        if args.if_pre == 1:
            logits = model(img)
            probs = F.softmax(logits, dim=1).cpu()
            sel_nodes = logits[torch.arange(len(targ_class)), targ_class]
            sel_nodes.backward(ones)
            logits = logits.cpu()

        else:
            probs = model(img)[0]
            sel_nodes = probs[torch.arange(len(targ_class)), targ_class]
            sel_nodes.backward(ones)
            probs = probs.cpu()

        grad = img.grad.cpu().numpy()  # [1, 3, 32, 32]
        grad = np.rollaxis(grad, 1, 4)  # [1, 32, 32, 3]
        # normalize to [0, 1]
        grad = grad[0]
        if 'imgs' in val_loader.dataset:
            img_name = val_loader.dataset.imgs[i][0].split('/')[-1]
            np.save(f'{out_path}/{img_name}.npy', grad)
        else:
            np.save(f'{out_path}/{i:05d}.npy', grad)


        if save_org:
            org = testset[i][0]
            org.save(f'{org_path}/{i:05d}.npy')
        pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    args = get_arguments()
    ############################################
    if args.dataset == 'imagenet':
        dataset = ImageNet('/home/chirag/convergent_learning/data')
    else:
        dataset = CIFAR('data')
        # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        #                                   download=True)


    model_path = args.model_path
    folder_name = model_path.split('/')[-2]
    out_path = f'result/grad_data/{folder_name}'
    check_path(out_path)
    save_grad(network=args.network, model_path=model_path, dataset=dataset, out_path=out_path,
              batch_size=args.batch_size, pytorch_pretrained=args.pytorch_model)

    img_path = f'result/grad_img/{folder_name}'
    check_path(out_path)
    plot_grad(out_path, img_path)





