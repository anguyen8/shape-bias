from loader.class_loader import alblation_load_model, loadmodel
from settings import alblation_setting as setting
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from utils import check_path
import pandas as pd

# variables for hooker
feature_before = []
feature_after  = []

# Map imagenet names to their netids
def image_name_to_netid():

    input_f = open("ILSVRC2012/imagenet_validation_imagename_labels.txt")
    label_map = {}
    netid_map = {}
    for line in input_f:
        parts = line.strip().split(" ")
        label_map[parts[0]] = parts[1]
        netid_map[parts[0]] = parts[2]
    return label_map, netid_map


class CustomDataset:
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label, self.netid = image_name_to_netid()

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ipdb.set_trace()
        img_name = os.path.join(self.root_dir, sorted(os.listdir(self.root_dir))[idx])
        trans = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

        image = trans(Image.open(img_name).convert('RGB'))
        val_img = image.unsqueeze(0)
        val_img = val_img.cuda()


        return val_img, self.label[img_name.split('/')[-1]], self.netid[img_name.split('/')[-1]], img_name


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        print(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def dissect_layers(model_name):
    if model_name in ['alexnet', 'AlexNet']:
        return ['1', '4', '7', '9', '11']


# hooker for features
def hook_feature_before(module, input, output):
    feature_before.append(output.data.cpu().numpy())


def hook_feature_after(module, input, output):
    feature_after.append(output.data.cpu().numpy())


def set_hookers(model_name, hooker_layer, model_before, model_after):
    if model_name == 'alexnet':
        for layer in hooker_layer:
            model_after._modules.get('features')._modules.get(layer).register_forward_hook(hook_feature_after)
            model_before._modules.get('features')._modules.get(layer).register_forward_hook(hook_feature_before)

def predict_and_save(data_path, correct_path, model_name=setting.MODEL, surgery_label= setting.TARGET_LABEL ,
                     zero_dict = None, save_path=False):
    """Lazy setting"""
    model_path = {'alexnet-r': 'zoo/alexnet-r.pt', 'alexnet':'zoo/alexnet.pth'}
    if model_name[-1] in ['r', 'R']:
        madry = True
    else:
        madry = False

    global feature_before
    global feature_after
    # load model
    model_surgical = alblation_load_model(model_name, model_path[model_name], madry, True, True, zero_dict)
    model_org      = loadmodel(model_name, model_path[model_name], madry, True, True)
    # set desire layer
    obs_layers = dissect_layers('alexnet')
    # register hooker
    set_hookers('alexnet', obs_layers, model_org, model_surgical)
    # load data index
    data_loader = CustomDataset(data_path)
    # empty list for stats
    diff_stats = []
    for i in range(len(obs_layers)):
        diff_stats.append([])
    with torch.no_grad():
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # evaluate images
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(f"Finding difference on {model_name} {surgery_label}")
        for i, (img, target, netid, img_name) in enumerate(data_loader):
            target_tensor = torch.tensor([int(target)]).to('cuda')


            # feed image to both model
            score_surgical = model_surgical(img)
            score_org      = model_org(img)

            for k in range(len(obs_layers)):
                diff_stats[k].append(feature_before[k][0]-feature_after[k][0])


            # prob = torch.nn.functional.softmax(score, dim=1)
            # (acc1, acc5), _ = accuracy(prob, target_tensor, (1, 5))
            # top1.update(acc1[0])
            # top5.update(acc5[0])

            # clear info in hooker for new image
            feature_before = []
            feature_after  = []
            # Display progress
            pbar.update(1)


        pbar.close()

    # save results as npy by layer
    for i, layer_name in enumerate(obs_layers):
        np.save(f'{save_path}/{layer_name}.npy', np.array(diff_stats[i]))



if __name__ == '__main__':
    img_path = setting.VAL_PATH
    correct_dir = 'dataset/correct'
    check_path(correct_dir)

    '''evaluate sets of surgery'''
    # networks = ['alexnet', 'alexnet-r']
    # labels = ["striped", 'zigzagged', 'chequered', 'banded', 'perforated']
    # for network in networks:
    #     for aim in labels:
    #         # print(f"Testing on {network} {aim}")
    #         zero_out = pd.read_csv(f"result/tally/alblation_csv/{network}/{aim}.csv", index_col=0, dtype=str,
    #                                keep_default_na=False).to_dict(orient='list')
    #         predict_and_save(img_path, correct_dir, network, aim, zero_out)
    '''evalue one layer'''
    network = 'alexnet-r'
    aim = 'orange-c'
    layer = '1'
    topk = '1'
    csv_name = f'{layer}_{aim}_{topk}.csv'
    zero_out = pd.read_csv(f"result/tally/alblation_csv/{network}/byLayer/{csv_name}", index_col=0, dtype=str,
                                   keep_default_na=False).to_dict(orient='list')
    out_put_path = f'result/observe_surgery/{network}'
    check_path(out_put_path)
    predict_and_save(img_path, correct_dir, network, aim, zero_out, save_path=out_put_path)