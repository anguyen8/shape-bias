from settings import settings
from loader.model_loader import loadrobust, loadmodel
from feature_operation import hook_feature, FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
# for loading robust version
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS
import torch
import numpy as np
# choose a random gpu
if torch.cuda.is_available():
    # gpu_counts = torch.cuda.device_count()
    # gpu_chosen = np.random.randint(gpu_counts)
    torch.cuda.set_device(settings.GPU_ID)
    print("Using GPU:", settings.GPU_ID)

fo = FeatureOperator()

if settings.MODEL[-1] in ['r', 'R']:
    '''loading robust model'''
    dataset = DATASETS['imagenet']('robustness/dataset')
    model, checkpoint = make_and_restore_model(arch=settings.MODEL[:-2],
                                               dataset=dataset, parallel=settings.MODEL_PARALLEL,
                                               resume_path=settings.MODEL_FILE)


    model = loadrobust(hook_feature, model, checkpoint,  settings.FEATURE_NAMES)
else:
    model = loadmodel(hook_feature)

############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    ############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features[layer_id], savepath="quantile"+str(layer)+".npy")

    ############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id], thresholds, savepath="tally"+str(layer)+".csv")

    ############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()
