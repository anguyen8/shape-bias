from settings import settings
from loader.model_loader import loadrobust
# for loading robust version
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS
import torch
from PIL import Image
from torchvision import transforms

# target layers can be changed in settings file. For this sample, it will hook layer ["1", "4", "7", "9", "11"] of AlexNet-R
features_blobs = []
def hook_feature(module, input, output): 
    features_blobs.append(output.data.cpu().numpy())

def sample_loading_and_hooking():
    global features_blobs
    if settings.MODEL[-1] in ['r', 'R']:
        '''loading robust model'''
        dataset = DATASETS['imagenet']('robustness/dataset')
        model, checkpoint = make_and_restore_model(arch=settings.MODEL[:-2],
                                                dataset=dataset, parallel=settings.MODEL_PARALLEL,
                                                resume_path=settings.MODEL_FILE)

        model = loadrobust(hook_feature, model, checkpoint,  settings.FEATURE_NAMES)

    img = Image.open('figures/real/ILSVRC2012_val_00002691.JPEG')

    TEST_TRANSFORMS_IMAGENET = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    input_tensor = TEST_TRANSFORMS_IMAGENET(img).unsqueeze(0).cuda()

    logit, _ = model(input_tensor)
    pred_class = logit.argmax()
    hooked_features = dict(zip(["1", "4", "7", "9", "11"], features_blobs)) # dict for features of layer1, 4, 7, 9, 11.
    return logit, pred_class, hooked_features

if __name__ == '__main__':
    #* A naive sample for load and hook features from robust model
    logit, pred_class, hooked_features = sample_loading_and_hooking()