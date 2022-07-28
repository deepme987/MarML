
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_frcnn_model(model_path=None, num_classes=0, pretrained=True, fine_tune=True):

    if model_path:
        assert os.path.exists(model_path), "Unable to find specified model"
        print('[INFO]: Loading model checkpoint:', model_path)
        model = torch.load(model_path)
        return model

    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
