
import utils
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms as T

from smd import SMDDataset
from frcnn import get_frcnn_model
from retinanet import get_retina_model
from efficientnet import get_eb_model, train_one_epoch_eb
from engine import train_one_epoch, evaluate

from variables import *

torch.manual_seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


MODEL_MAP = {
    "FRCNN": {
        "model": get_frcnn_model
    },
    # "EB": {
    #     "model": get_eb_model
    # },
    "Retina": {
        "model": get_retina_model
    }
}


"""
# Possible transformation
import torchvision.transforms.functional as TF
import random

def my_rotation(image, bonding_box_coordinate):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        bonding_box_coordinate = TF.rotate(bonding_box_coordinate, angle)
    # more transforms ...
    return image, bonding_box_coordinate
"""


def get_transform(train):
    transforms = [T.ToTensor()]
    # Breaks due to no transform on bounding box
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_model(architecture="FRCNN", model_checkpoint=None, eval_only=False):
    print("------------------Initializing Training------------------")
    print("Using device:", device)

    data = SMDDataset(DATA_DIR, transform=get_transform(True))

    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)

    # data_loader, data_loader_test = utils.get_split_dataset(data, split_ratio=TRAIN_TEST_SPLIT, batch_size=BATCH_SIZE)

    # our dataset has two classes only - background and person
    num_classes = NUM_CLASSES

    # TODO: Add this based on variable (maybe dict mapper?)

    model = MODEL_MAP[architecture]["model"](model_path=model_checkpoint, num_classes=num_classes,
                                             pretrained=True, fine_tune=True)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Start with higher LR and drop with increase in epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = EPOCHS

    if eval_only:
        evaluate(model, data_loader, device=device)
        return

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        torch.save(model, f"models/{architecture}-{epoch}.pth")


if __name__ == '__main__':
    train_model(architecture=MODEL_NAME, model_checkpoint=MODEL_CHECKPOINT, eval_only=EVAL_FLAG)
