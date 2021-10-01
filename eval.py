from loader.class_loader import *
from settings import eval_setting
import torch
import os
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


def eval_by_class():
    model = loadmodel()
    labelMap = load_imagenet_label_map()
    data_path = eval_setting.VAL_PATH

    class_hist = pd.DataFrame(columns=['Class', 'Top1', 'Top5'])
    class_hist.Class = labelMap.Name
    class_confidence = pd.DataFrame(columns=['Class', 'Confidence', 'Images', 'Confidence5', 'Images5'])
    class_confidence.Class = labelMap.Name
    progress = ProgressMeter(
                            1000,
                            prefix='Val: ')
    for i in range(1000):
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        imageList, class_name = labelMap.loc[i, 'images'].split(","), labelMap.loc[i, "Name"]
        target = torch.Tensor([i])
        confidence = ""
        images = ""
        confidence5 = ""
        images5 = ""
        with torch.no_grad():
            #evaluate images
            for img in imageList:
                filepath = os.path.join(data_path, img)
                val_img = load_val(filepath)
                score = model(val_img)
                prob = torch.nn.functional.softmax(score, dim=1)
                acc1, acc5 = accuracy(prob, target, (1, 5))
                top1.update(acc1[0])
                top5.update(acc5[0])
                # save confidence if correctly classified
                if int(acc1):
                    if confidence == "":
                        confidence = str(float(prob.max()))
                    else:
                        confidence += ','+str(float(prob.max()))
                    if images == "":
                        images = img
                    else:
                        images += ',' + img
                if int(acc5):
                    if confidence5 == "":
                        confidence5 = str(float(prob.max()))
                    else:
                        confidence5 += ','+str(float(prob.max()))
                    if images5 == "":
                        images5 = img
                    else:
                        images5 += ',' + img
            # Display progress
            print('Class **'+class_name+'** Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%'
                  .format(top1=top1, top5=top5))
            progress.display(i+1)
            # update classify results
            class_hist.loc[i, 'Top1'] = float(top1.avg)
            class_hist.loc[i, 'Top5'] = float(top5.avg)
            class_confidence.loc[i, 'Confidence'] = confidence
            class_confidence.loc[i, 'Images']     = images
            class_confidence.loc[i, 'Confidence5'] = confidence5
            class_confidence.loc[i, 'Images5']     = images5


    # save results
    result_path = os.path.join("result", str.lower(eval_setting.MODEL))
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    class_hist.to_csv(result_path+"/hist.csv")
    class_confidence.to_csv(result_path + "/confidence.csv")

if __name__ == '__main__':
    eval_by_class()