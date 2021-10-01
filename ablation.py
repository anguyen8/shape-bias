from loader.class_loader import *
from settings import alblation_setting as setting
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import numpy as np
from loader.target.find_zero_unit import find_zero_dict
from utils import check_path, str2bool
import torchvision.datasets as datasets
import ipdb

def image_name_to_netid():
    # Map imagenet names to their netids
    input_f = open("ILSVRC2012/imagenet_validation_imagename_labels.txt")
    label_map = {}
    netid_map = {}
    for line in input_f:
        parts = line.strip().split(" ")
        label_map[parts[0]] = int(parts[1])
        netid_map[parts[0]] = parts[2]
    return label_map, netid_map


# class CustomDataset:
#     """Face Landmarks dataset."""
#
#     def __init__(self, root_dir, transform=None):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.label, self.netid = image_name_to_netid()
#
#     def __len__(self):
#         return len(os.listdir(self.root_dir))
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         # ipdb.set_trace()
#         img_name = os.path.join(self.root_dir, sorted(os.listdir(self.root_dir))[idx])
#         trans = transforms.Compose([transforms.Resize(256),
#                                     transforms.CenterCrop(224),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                          std=[0.229, 0.224, 0.225])
#                                     ])
#
#         image = trans(Image.open(img_name).convert('RGB'))
#         # image = image.unsqueeze(0)
#         val_img = image.cuda()
#
#
#         return val_img, self.label[img_name.split('/')[-1]], self.netid[img_name.split('/')[-1]], img_name

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


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters=None, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix
#
#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         print(entries)
#
#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def predict_and_save(data_path, model_name=setting.MODEL, surgery_label= setting.TARGET_LABEL, inter_list=None,
                     zero_dict = None, csv_name="", correctPath='dataset/correct_csv_alblation',
                     wrongPath='dataset/wrong_csv_alblation', keep=False, resize=True, normalizing=True):
    """Lazy setting"""
    model_path = {'alexnet-r': 'zoo/alexnet-r.pt', 'alexnet': 'zoo/alexnet.pth',
                  'resnet50': 'zoo/ResNet50.pt', 'resnet50-r': 'zoo/ResNet50_R.pt'
                    }
    if model_name[-1] in ['r', 'R']:
        madry = True
    else:
        madry = False

    # load model
    model = alblation_load_model(model_name, model_path[model_name], madry, True, True, zero_dict, keep)
    # load data index
    # data_loader = CustomDataset(data_path)
    # dataloader = DataLoader(data_loader, batch_size=batch_size)

    #dataloader
    batch_size = 32

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

        predict_csv = pd.DataFrame(columns=['File', 'Confidence', 'Class'])
        error_csv = pd.DataFrame(columns=['File', 'Confidence', 'Class'])
        # evaluate images
        # pbar = tqdm(total=len(val_loader))
        pbar = tqdm(total=len(dataloader))
        pbar.set_description(f"Testing on {model_name} {surgery_label}")

        if inter_list is not None:
            intersection = True
        else:
            intersection = False
        for i, (img, target) in enumerate(dataloader):
        # for img, target, netid, img_name in dataloader:
        #     img_name = pd.DataFrame(dataloader.dataset.imgs[i * batch_size: (i + 1) * batch_size], columns=['File', 'Class'])
            # ipdb.set_trace()
            if intersection:
                img_name = dataloader.dataset.imgs[i][0].split('/')[-1]
                if img_name in inter_list:
                    img = img.to('cuda')
                    target_tensor = torch.tensor(list(target)).to('cuda')
                    score = model(img)
                    prob = torch.nn.functional.softmax(score, dim=1)
                    (acc1, acc5), correct_top1 = accuracy(prob, target_tensor, (1, 5))
                    top1.update(acc1[0])
                    top5.update(acc5[0])

                    # save images to DataFrame
                    batch_data = pd.DataFrame([img_name, prob.max(axis=1).values.tolist(), target.tolist()]).T
                    batch_data.columns=['File', 'Confidence', 'Class']
                    predict_csv = predict_csv.append(batch_data[correct_top1], ignore_index=True)
                    wrong_batch_list = ~np.array(correct_top1)
                    error_csv = error_csv.append(batch_data[wrong_batch_list], ignore_index=True)
            else:
                img = img.to('cuda')
                target_tensor = torch.tensor(list(target)).to('cuda')
                score = model(img)
                prob = torch.nn.functional.softmax(score, dim=1)
                (acc1, acc5), correct_top1 = accuracy(prob, target_tensor, (1, 5))
                top1.update(acc1[0])
                top5.update(acc5[0])

                # save images to DataFrame
                # for imagenet layout
                batch_data = pd.DataFrame(dataloader.dataset.imgs[i * batch_size: (i + 1) * batch_size], columns=['File', 'Class'])
                batch_data['Confidence'] = prob.max(axis=1).values.tolist()
                # batch_data = pd.DataFrame([img_name, prob.max(axis=1).values.tolist(), target.tolist()]).T
                # batch_data.columns = ['File', 'Confidence', 'Class']
                predict_csv = predict_csv.append(batch_data[correct_top1], ignore_index=True)
                wrong_batch_list = ~np.array(correct_top1)
                error_csv = error_csv.append(batch_data[wrong_batch_list], ignore_index=True)
            # Display progress
            pbar.update(1)
        pbar.close()

    # save results as csv
    # correct prediction
    if csv_name == "":
        csv_name = surgery_label+".csv"
    predict_csv.to_csv(correctPath+"/"+csv_name, index=True)
    # wrong prediction
    error_csv.to_csv(wrongPath+"/"+csv_name, index=True)
    del predict_csv, error_csv

    print(f'{model_name} {surgery_label}' + ': Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'
          .format(top1=top1, top5=top5))
    return '{top1.avg:.3f}%'.format(top1=top1)

# argument parser
def argParser():
    parser = argparse.ArgumentParser(description='Evaluate surgery 1 neuron at a time')
    parser.add_argument('--network', default=['alexnet', 'alexnet-r'],  help='Model name')
    parser.add_argument('--range_s', default=0, type=int, metavar='N', help='staring point of surgery')
    parser.add_argument('--range_e', default=1, type=int, metavar='N', help='End point of surgery')
    parser.add_argument('--layers', help='Target layers', type=list, nargs='+', default=['1'])
    parser.add_argument('--labels', help='Target label', type=list, nargs='+', default=['whatever'])
    parser.add_argument('--topk', default='1', help='Number of target channels')
    parser.add_argument('--gpu', default=0, type=int, metavar='N', help='Run on which gpu')
    parser.add_argument('--alb_type', default='zero_out', help='Keep the target neuron or zero out the target')
    parser.add_argument('--dataset', default='/home/chirag/convergent_learning/data/val/', help='dataset for testing')
    parser.add_argument('--data_list', default=None, help='Restrict data to be a specific set')
    parser.add_argument('--save_folder', default='surgery_temp',  help='Model name')
    parser.add_argument('--split_job', default='1.1', help='Split job=a.b, means split into a parts, and run for part b. ')
    parser.add_argument('--single_concept', default=True, type=str2bool,  help='zero_out/keep single concept or not ')
    parser.add_argument('--resize', default=True, type=str2bool, help='resize or not')
    parser.add_argument('--normalize', default=True, type=str2bool, help='normalize or not')
    args = parser.parse_args()

    # set to deal with multiple layers
    layers = args.layers
    for idx, layer_item in enumerate(layers):
        layers[idx] = ''.join(layers[idx])
    args.layers = layers
    # if input is labels, return labels, otherwise load labels
    target_label = args.labels
    for idx, label_item in enumerate(target_label):
        target_label[idx] = ''.join(target_label[idx])
    args.labels = target_label
    if os.path.isfile(args.labels[0]):
        args.labels = pd.read_csv(args.labels[0]).Concept.to_list()

    if not isinstance(args.network, list):
        args.network = args.network.split('_')

    if args.data_list is not None:
        if os.path.isfile(args.data_list):
            args.data_list = pd.read_csv(args.data_list).File.to_list()

    return args

def alblationTest(args):
    img_path = args.dataset
    surgery_start = args.range_s
    surgery_end = args.range_e
    layers = args.layers
    target_label = args.labels
    gpu_id = args.gpu
    inter_list = args.data_list
    if args.topk in ['same', 'all']:
        topk = args.topk
    else:
        topk = int(args.topk)
    save_folder = args.save_folder
    alblation_type = args.alb_type
    networks = args.network
    if args.alb_type == 'zero_out':
        keep_target = False
    else:
        keep_target = True

    idx_range = surgery_end - surgery_start
    torch.cuda.set_device(gpu_id)
    accuracy_record = pd.DataFrame(columns=networks)
    for net_idx, network in enumerate(networks):
        correct_path = f'result/correct_csv_alblation/{alblation_type}/{network}/{save_folder}'
        wrong_path = f'result/wrong_csv_alblation/{alblation_type}/{network}/{save_folder}'
        check_path(correct_path)
        check_path(wrong_path)

        # zero out with single layer
        if len(layers) == 1:
            layer = layers[0]
            # zero out channels in target layer by channel ID
            if target_label[0] == 'whatever':
                accuracy_record = pd.DataFrame(columns=networks, index=range(idx_range))
                net_stats = pd.read_csv(f'result/tally/{network}/tally{layer}.csv', index_col=False)
                net_stats.sort_values(by='unit', inplace=True)
                net_stats = net_stats.reset_index(drop=True)
                for neuron_id in range(surgery_start, surgery_end):
                    # table that save accuracy results
                    # zero out dictionary
                    zero_out = {layer: [f'{neuron_id}']}
                    label_name = net_stats.loc[neuron_id, 'label']
                    acc = predict_and_save(img_path, network, neuron_id, zero_dict=zero_out, inter_list=None,
                                           csv_name=f'{str(neuron_id).zfill(3)}_{label_name}.csv', correctPath=correct_path, wrongPath=wrong_path,
                                           keep=keep_target, resize=args.resize, normalizing=args.normalize)
                    accuracy_record.loc[neuron_id, network] = acc
                    del zero_out
            # zero out channels in target layer with target labels
            else:
                # accuracy_record = pd.DataFrame()
                if topk == 'same':
                    zero_list = find_zero_dict(networks, target_label, [layer], topk='same')
                    zero_out = zero_list[net_idx]

                else:
                    zero_out = find_zero_dict(network, target_label, [layer], topk=topk)
                label = '_'.join(target_label)
                acc = predict_and_save(img_path, network, label, inter_list, zero_out, f'{layer}_{label}_{topk}.csv',
                                       correct_path, wrong_path, keep_target, resize=args.resize, normalizing=args.normalize)
                accuracy_record.loc[target_label[0], network] = acc

        # zero out across all target layers
        else:
            # accuracy_record = pd.DataFrame()
            label = '_'.join(target_label)
            # zero all all channels for target label
            if topk == 'all':
                zero_out, channel_count = find_zero_dict(network, target_label, layers, topk=topk, channel_cout=True)
                acc = predict_and_save(img_path, network, label, zero_dict=zero_out, inter_list=None,
                                       csv_name=f'{label}_all.csv', correctPath=correct_path, wrongPath=wrong_path,
                                       keep=keep_target, resize=args.resize, normalizing=args.normalize)
                accuracy_record.loc[target_label[0], network] = acc
            # zero out same amount of channels for target labels
            elif topk == 'same':
                # accuracy_record = pd.DataFrame()
                zero_list = find_zero_dict(networks, target_label, [layer], topk='same')
                zero_out = zero_list[net_idx]
                acc = predict_and_save(img_path, network, label, zero_dict=zero_out, inter_list=None,
                                       csv_name=f'{label}_{channel_count}.csv', correctPath=correct_path, wrongPath=wrong_path,
                                       keep=keep_target, resize=args.resize, normalizing=args.normalize)
                accuracy_record.loc[target_label[0], network] = acc
            # zero out top k channel in each target layer
            else:
                zero_out, channel_count = find_zero_dict(network, target_label, layers, topk=topk, channel_cout=True)
                acc = predict_and_save(img_path, network, label, zero_dict=zero_out, inter_list=None,
                                       csv_name=f'{label}_{channel_count}.csv', correctPath=correct_path, wrongPath=wrong_path,
                                       keep=keep_target, resize=args.resize, normalizing=args.normalize)
            accuracy_record.loc[target_label[0], network] = acc
    accuracy_record_path = f'result/zero_out_acc/{network}/{alblation_type}/{save_folder}'
    check_path(accuracy_record_path)
    accuracy_record.to_csv(f'{accuracy_record_path}/{save_folder}.csv', index=True)



if __name__ == '__main__':
    # read arguments
    args = argParser()
    # run script for all unique concepts
    if len(args.labels) > 1 and args.single_concept:
        labels = args.labels

        # split job into multiple jobs
        jobs =int(args.split_job.split('.')[0])
        if jobs > 1:
            import math
            job_length = math.ceil(len(labels)/jobs)
            work_on = int(args.split_job.split('.')[1])
            labels = labels[(work_on-1)*job_length: work_on*job_length]
        # no labels for the worker
        if len(labels) == 0:
            exit(0)

        for label in labels:
            args.labels = [label]
            alblationTest(args)
    else:
        alblationTest(args)

