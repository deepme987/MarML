
import os
import sys
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import utils


def get_retina_model(model_path=None, num_classes=0, pretrained=True, fine_tune=True):
    if model_path:
        assert os.path.exists(model_path), "Unable to find specified model"
        print('[INFO]: Loading model checkpoint:', model_path)
        model = torch.load(model_path)
        return model

    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # replace classification layer
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits

    return model
