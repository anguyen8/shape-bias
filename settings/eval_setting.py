VAL_PATH    = "dataset/correct/alexnet_inter"  # validation dataset
TEXTURE_PATH = "imagenet_texture_bb"
CORRECT_PATH = "dataset/correct"
ROBUST      = False
MADRYMODEL  = False
MODEL       = 'alexnet-r'                          # model arch:  alexnet, resnet50, googlenet
MODEL_PATH  = 'zoo/alexnet-r.pt'                   # resume model path for robust model
DATASET     = 'imagenet'                              #  model trained on: places365 or imagenet


"""setting for GoogleNet"""
AUXILIARY = False                                     # whether Auxiliary layer are in used
# Model setting
if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
# ResNet50
if MODEL == 'resnet50':
    MODEL_FILE = '../zoo/ResNet50.pt'
    MODEL_PARALLEL = True
elif MODEL == 'resnet50-r':
    MODEL_FILE = '../zoo/ResNet50_R.pt'
    MODEL_PARALLEL = True
# GoogLeNet
elif MODEL == 'googlenet':
    MODEL_FILE = 'zoo/GoogleNet.pt'
    MODEL_PARALLEL = False
elif MODEL == 'googlenet-r':
    MODEL_FILE = '../zoo/GoogleNet_R.pt'
    MODEL_PARALLEL = False
# AlexNet
elif MODEL == 'alexnet':
    MODEL_FILE = 'zoo/alexnet.pth'
    MODEL_PARALLEL = True
elif MODEL == 'alexnet-r':
    MODEL_FILE = 'zoo/alexnet-r.pt'
    MODEL_PARALLEL = True

WORKERS = 12
BATCH_SIZE = 128
TALLY_BATCH_SIZE = 16
TALLY_AHEAD = 4
INDEX_FILE = 'index.csv'
GPU       = True                                    # running on GPU is highly suggested
TEST_MODE = False                                   # turning on the testmode means the code will run on a small dataset.
CLEAN     = True