import torch
import torch.nn as nn
from settings import alblation_setting

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, zero_dict=None, keep=False):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.zo_dict = zero_dict
        self.keep = keep

    def forward(self, x):
        # zo_dict = alblation_setting.ZERO_OUT
        target_layers = self.zo_dict.keys()
        for i, layer in enumerate(self.features):
            # zero out values
            x = self.features[i](x)
            if str(i) in target_layers:
                zero_tensor = torch.zeros_like(x[0, 0]).cuda()
                # zero out target channels
                if not self.keep:
                    for unit in self.zo_dict[str(i)][0].split(","):
                        if unit == '':
                            continue
                        x[:, int(unit), ::] = zero_tensor
                # keep target channels and zero out the rest
                else:
                    for unit in range(x.shape[1]):
                        if str(unit) not in self.zo_dict[str(i)][0].split(","):
                            x[:, int(unit), ::] = zero_tensor

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)

    return model