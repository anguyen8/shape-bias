from settings import settings
import torch
import torchvision
from loader.models import alexnet

def loadmodel(hook_fn, network=settings.MODEL, model_path=settings.MODEL_FILE, layers=settings.FEATURE_NAMES):
    if model_path is None:
        model = torchvision.models.__dict__[network](pretrained=True)
    else:
        checkpoint = torch.load(model_path)
        if type(checkpoint).__name__ in ['OrderedDict', 'dict']:
            if network[0:6] == 'google':
                model = torchvision.models.__dict__[network](num_classes=settings.NUM_CLASSES,
                                                                    aux_logits=settings.AUXILIARY)
            elif network in ['resnet50-sin', 'resnet50-sin-in', 'resnet50-ft']:
                model = torchvision.models.__dict__['resnet50'](num_classes=settings.NUM_CLASSES)
            else:
                model = torchvision.models.__dict__[network](num_classes=settings.NUM_CLASSES)
                '''For zero out test'''
                # model = alexnet.AlexNet()
            if settings.MODEL_PARALLEL:
                if network in ['resnet50-sin', 'resnet50-sin-in', 'resnet50-ft']:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
                else:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

    if (layers[0] != "features") and (network[0:4] == 'alex'):
        # load alexnet layer by layer
        for layer in layers:
            model._modules.get(settings.LAYER_SECTION)._modules.get(layer).register_forward_hook(
                hook_fn)
    else:
        for name in layers:
            model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model



def loadrobust(hook_fn, model, checkpoint,  layers, network=settings.MODEL, parallel=settings.MODEL_PARALLEL):

    # load stat dict
    if type(checkpoint).__name__ in ['OrderedDict', 'dict']:
        # the data parallel layer will add 'module' before each layer name
        # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}

        # for googlenet
        if (
            str.lower(network) == 'googlenet'
            or str.lower(network) != 'googlenet'
            and str.lower(network) == 'googlenet-r'
        ):
            state_dict = checkpoint['model']
            if str.lower(network)[0:9] == 'googlenet':
                state_dict = {str.replace(k, 'branch4_conv', 'branch4.1'): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint['model']
        if not parallel:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    else:
        model = checkpoint

    # load feature output
    if str.lower(network)[0:9] == 'googlenet':
        for layer in layers:
            if parallel:
                model._modules.get('module')._modules.get('model')._modules.get(layer).register_forward_hook(hook_fn)
            else:
                model._modules.get('model')._modules.get(layer).register_forward_hook(hook_fn)

    elif str.lower(network)[0:7] == 'alexnet':
        layer_section = 'features'
    # for inspecting a section as a whole, change FEATURE_NAME in settings
        if parallel:
            if len(layers) == 0:
                model._modules.get('module').model._modules.get(layer_section).register_forward_hook(hook_fn)
            # choose specific layers in a section
            else:
                for layer in layers:
                    model._modules.get('module').model._modules.get(layer_section)._modules.get(layer).register_forward_hook(
                        hook_fn)
        elif len(layers) == 0:
            model._modules.get('model')._modules.get(layer_section).register_forward_hook(hook_fn)
        else:
            for layer in layers:
                model._modules.get('model')._modules.get(layer_section)._modules.get(layer).register_forward_hook(
                    hook_fn)

    else:
        for layer in layers:
            if parallel:
                model._modules.get('module')._modules.get('model')._modules.get(layer).register_forward_hook(
                    hook_fn)
            else:
                model._modules.get('model')._modules.get(layer).register_forward_hook(hook_fn)


    if settings.GPU:
        model.cuda()
    model.eval()
    return model

