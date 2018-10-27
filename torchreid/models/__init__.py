from __future__ import absolute_import

from .resnet import *
from .resnetmid import *
from .senet import *
from .mudeep import *
from .hacnn import *
from .mobilenetv2 import *
from .shufflenet import *

"""
from .resnext import *
from .densenet import *
from .mudeep import *
from .hacnn import *
from .squeeze import *
from .xception import *
from .inceptionv4 import *
from .nasnet import *
from .inceptionresnetv2 import *
"""


__model_factory = {
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
    'resnet50mid': resnet50mid,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    #'resnext101': ResNeXt101_32x4d,
    #'densenet121': DenseNet121,
    #'squeezenet': SqueezeNet, # https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
    'mobilenetv2': MobileNetV2,
    'shufflenet': ShuffleNet,
    #'xception': Xception,
    #'inceptionv4': InceptionV4,
    #'nasnsetmobile': NASNetAMobile,
    #'inceptionresnetv2': InceptionResNetV2,
    'mudeep': MuDeep,
    'hacnn': HACNN,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)