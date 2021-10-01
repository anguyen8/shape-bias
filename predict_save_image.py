from loader.class_loader import *
from settings import eval_setting
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os, shutil, argparse
from tqdm import tqdm
import numpy as np
from utils import check_path, scrimble_img
import torchvision.datasets as datasets_torch
from robustness import datasets
from robustness.datasets import CIFAR, ImageNet

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

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


def predict_and_save(data_path, correct_path, model_name, model_path, Madry_model=False, data_type=None, resize=True, normalizing=True):

    if args.pytorch_model:
        import torchvision.models as models
        if model_name == 'googlenet':
            model = models.googlenet(pretrained=True)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        model.cuda()
        model.eval()
    else:
        model = loadmodel(model_name, model_path, Madry_model)
    batch_size = 128

    if args.madry_setting:
        dataset = ImageNet('/home/peijie/ILSVRC2012')
        train_loader, val_loader = dataset.make_loaders(batch_size=batch_size, workers=8)
        dataloader = val_loader
    else:
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
            datasets_torch.ImageFolder(data_path, transforms.Compose(trans)),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)



    with torch.no_grad():
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        predict_csv = pd.DataFrame(columns=['File', 'Confidence', 'Class'])

        # batch_size = 1
        # dataloader = DataLoader(data_loader, batch_size=batch_size)

        # save path
        correct_img_path = os.path.join(correct_path, eval_setting.MODEL)
        check_path(correct_img_path)
        # evaluate images

        # for i, (img, target, netid, img_name) in enumerate(data_loader):
        # target = torch.tensor([int(target)]).to('cuda')
        pbar = tqdm(total=len(dataloader))
        pbar.set_description(f"Classifying {model_name}")
        for i, (img, target) in enumerate(dataloader):
        # for img, target, netid, img_name in dataloader:
            target_tensor = torch.tensor(list(target)).to('cuda')
            img = img.to('cuda')
            logit = model(img)
            # Compute confidence
            prob = torch.nn.functional.softmax(logit, dim=1)
            (acc1, acc5) = accuracy(prob, target_tensor, (1, 5))

            # Confidence not needed
            # (acc1, acc5), correct_top1 = accuracy(score, target_tensor, (1, 5))

            top1.update(acc1[0])
            top5.update(acc5[0])

            # save confidence if correctly classified
            # correct_file = np.array(img_name)[correct_top1].tolist()
            batch_data = pd.DataFrame(dataloader.dataset.imgs[i * batch_size: (i + 1) * batch_size],
                                      columns=['File', 'Class'])
            batch_data['Confidence'] = prob.max(1).values.tolist()
            # predict_csv = predict_csv.append(batch_data[correct_top1], ignore_index=True)
            # Display progress
            pbar.update(1)
        pbar.close()

    # save results as csv
    # csv_path = os.path.join(correct_path.split("/")[0], "correct_csv")
    # check_path(csv_path)
    # if data_type is not None:
    #     csv_name = f'{model_name}_{data_type}.csv'
    # else:
    #     csv_name = f"{model_name}.csv"
    # pd.DataFrame(predict_csv).to_csv(csv_path+"/"+csv_name, index=True)

    print(f'{model_name}: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'
          .format(top1=top1, top5=top5))

def argParser():
    parser = argparse.ArgumentParser(description='Evaluate surgery 1 neuron at a time')
    parser.add_argument('--network', default='alexnet',  help='Model name')
    parser.add_argument('--model_path', default='zoo/alexnet.pth', help='Model path')
    parser.add_argument('--data_path', default='/home/peijie/ILSVRC2012/val', help='path to dataset')
    parser.add_argument('--madry', action='store_true', help='weather it is a madry model')
    parser.add_argument('--data_type', default='clean', help='Model name')
    parser.add_argument('--resize', default=True, type=bool, help='resize or not')
    parser.add_argument('--normalize', default=True, type=bool, help='normalize or not')
    parser.add_argument('--out_path', default='dataset/correct', help='output path')
    parser.add_argument('--madry_setting', action='store_true')
    parser.add_argument('--pytorch_model', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argParser()
    # img_path = '/home/chirag/convergent_learning/data/val/'
    # img_path = 'dataset/gaussian_noise'
    # img_path = '/home/chirag/stylized_imagenet/val'
    # img_path = '/home/chirag/convergent_learning/scrambling_dataset_112'
    check_path(args.out_path)
    predict_and_save(args.data_path, args.out_path, args.network, args.model_path, Madry_model=args.madry,
                     data_type=args.data_type, resize=args.resize, normalizing=args.normalize)
