import skimage
import skimage.feature
import skimage.viewer
import numpy as np
import argparse
import ipdb
import os
from skimage import img_as_uint, img_as_ubyte

def getArgs():
    parser = argparse.ArgumentParser(description = 'PyTorch ImageNet validation')
    parser.add_argument('--imagenetpath', metavar='PATH', default='ILSVRC2012/ILSVRC2012_img_val',
                        help='path to dataset')
    parser.add_argument('--inter_file', default='data/intersection_alexnet_alexnet_r.npy',
                        help='list of intersect images')
    parser.add_argument('--save_path', metavar='PATH', required=True,
                        help='save path for output images')
    parser.add_argument('--threshold_low', default=0.35, help='threshold of edge map')
    parser.add_argument('--threshold_high', default=0.9, help='threshold of edge map')
    parser.add_argument('--sigma', default=0.4, help='sigma of edge map')
    return parser.parse_args()


def generate_edge_map(args):
    os.makedirs(args.save_path, exist_ok=True)
    image_list = np.load(args.inter_file)
    for image_name in image_list:
        image = skimage.io.imread(f'{args.imagenetpath}/{image_name}', as_gray=True)
        edges = skimage.feature.canny(
            image=image,
            sigma=args.sigma,
            low_threshold=args.threshold_low,
            high_threshold=args.threshold_high,
        )
        skimage.io.imsave(f'{args.save_path}/{image_name}', img_as_ubyte(edges), check_contrast=False)
    

if __name__ == '__main__':
    args = getArgs()
    generate_edge_map(args)

