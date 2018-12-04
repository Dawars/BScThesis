"""
Calculate the number of tunable parameters in each encoder model
"""
import numpy as np
import torch
from torchvision import models

import Resnet, densenet


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    inception_model = models.inception_v3(pretrained=True)
    inception_layers = list(inception_model.children())
    del inception_layers[13]

    inception_layers[-1] = torch.nn.AvgPool2d(35, stride=1)
    encoder = torch.nn.Sequential(*inception_layers)

    print(f"inception {get_num_params(encoder)}")

    print(f"resnet {get_num_params(Resnet.load_Res50Model())}")

    print(f"densenet {get_num_params(densenet.load_denseNet('densenet121'))}")
