######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
GPU_ID = 0
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = True                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'alexnet-r'                          # model arch: resnet18, alexnet, resnet50, densenet161
# MODEL_PATH = 'zoo/alexnet-r.pt'           # resume model path for robust model
DATASET = 'imagenet'                       # model trained on: places365 or imagenet
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
LAYER_SECTION = None
CATAGORIES = ["object", "part", "scene", "texture", "color", "material"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
OUTPUT_FOLDER = "result/"+MODEL+"_"+DATASET # result will be stored in this folder
MADRYMODEL = True
"""setting for GoogleNet"""
AUXILIARY = True                           # whether Auxiliary layer are in used
########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL != 'alexnet' or 'alexnet-r':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
# ResNet18
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = '../zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False

# ResNet50
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    MODEL_FILE = 'zoo/ResNet50.pt'
    MODEL_PARALLEL = False
elif MODEL == 'resnet50-r':
    FEATURE_NAMES = ['layer3', 'layer4']
    MODEL_FILE = 'zoo/ResNet50_R.pt'
    MODEL_PARALLEL = False
elif MODEL == 'resnet50-sin':
    FEATURE_NAMES = ['layer3', 'layer4']
    MODEL_FILE = 'zoo/resnet50_SIN.pth.tar'
    MODEL_PARALLEL = True
elif MODEL == 'resnet50-sin-in':
    FEATURE_NAMES = ['layer4']
    MODEL_FILE = 'zoo/resnet50_sin_in.pth.tar'
    MODEL_PARALLEL = True
elif MODEL == 'resnet50-ft':
    FEATURE_NAMES = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    MODEL_FILE = 'zoo/resnet50_ft.pth.tar'
    MODEL_PARALLEL = True

# AlexNet
elif MODEL == 'alexnet':
    LAYER_SECTION = 'features'
    FEATURE_NAMES = ["1", "4", "7", "9", "11"]
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/alexnet.pth'
        MODEL_PARALLEL = False
elif MODEL == 'alexnet-r':
    LAYER_SECTION = 'features'
    FEATURE_NAMES = ["1", "4", "7", "9", "11"]
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/alexnet-r.pt'
        MODEL_PARALLEL = False

# GoogleNet
elif MODEL == 'googlenet':
    FEATURE_NAMES = ["conv1", "conv2", "conv3", "inception3a", "inception3b", "inception4a", "inception4b",
                     "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/GoogleNet.pt'
        MODEL_PARALLEL = True
elif MODEL == 'googlenet-r':
    FEATURE_NAMES = ["inception4b"]
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/GoogleNet_R.pt'
        MODEL_PARALLEL = True

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 12
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
