import os
import cv2
import glob

import torch
import numpy as np
from torch.utils.data import Dataset

from PIL import Image
from scipy.io import loadmat


"""
SMD Dataset ground truth reference:

cat_names = {
    1: "Ferry",
    2: "Buoy",
    3: "Vessel/ Ship",
    4: "Speed Boat",
    5: "Boat",
    6: "Kayak",
    7: "Sail Boat",
    8: "Swimming Person",
    9: "Flying object",
    10: "Other"
}
position_names = {
    1: "Moving",
    2: "Stationary",
    3: "Other"
}
distance_names = {
    1: "Near",
    2: "Far",
    3: "Other"
}
"""


def smd_to_yolo(file_path):
    """
    Convert SMD dataset to yolo format - reads video frames and their annotations from given SMD directory

    This creates a new directory NIR/data_yolo/ containing X frames with X .txt files for annotations

    For SMD dataset, we have 3 detection heads (3 separate OD labels)
    To make this work with single target, change the output.append below

    TODO: Find a way to dynamically handle this?
        Notes: Perhaps not feasible since it involves reading dataset in it's original format - which is unknown
    Args:
        file_path: path to the directory to read data from

    Returns:
        None: Extracts SMD videos and their annotations and converts into yolo format

    Usage: smd_to_yolo("NIR/")
    """
    object_dir = file_path + "/ObjectGT/"
    video_dir = file_path + "/Videos/"
    yolo_dir = file_path + "/data_yolo/"

    if not os.path.exists(yolo_dir):
        os.mkdir(yolo_dir)

    for fil in os.listdir(object_dir):

        data = loadmat(object_dir + fil)
        image_name = "_".join(fil.split("_")[:-1])
        image_dir = video_dir + image_name + ".avi"

        vidcap = cv2.VideoCapture(image_dir)
        success, image = vidcap.read()
        height, width, channels = image.shape
        count = 0
        while success:
            output = []
            position = data['structXML'][0][count][0]
            category = data['structXML'][0][count][1]
            distance = data['structXML'][0][count][2]

            bboxes = data['structXML'][0][count][6]

            for i in range(len(bboxes)):
                bbox = bboxes[i]
                try:
                    x, y, w, h = map(int, bbox)
                    x, y = (x + w) / 2, (y + h) / 2
                    output.append(
                        [position[i][0], category[i][0], distance[i][0], x / width, y / height, w / width, h / height])
                except:     # TODO: trace this exception
                    print(bbox)
                    print(count)
                    print(image_dir)
                    print(data['structXML'][0][count])
            cv2.imwrite(yolo_dir + image_name + "_" + str(count) + ".jpg", image)
            with open(yolo_dir + image_name + "_" + str(count) + ".txt", 'w+') as f:
                f.write("\n".join([" ".join(map(str, out)) for out in output]))
            success, image = vidcap.read()
            count += 1
        print(f"Extracted {count} frames")


class SMDDataset(Dataset):
    """
    Custom Dataset class for SMD dataset
    Reads images directly from given yolo directory
    """
    def __init__(self, yolo_dir, transform=None, target_transform=None):
        self.yolo_dir = yolo_dir
        self.images_dir = glob.glob(yolo_dir + "*.jpg")
        self.labels_dir = glob.glob(yolo_dir + "*.txt")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_dir)

    @staticmethod
    def read_yolo(label_path):
        with open(label_path, 'r') as f:
            data = f.readlines()
        data = [dat.split() for dat in data]
        data = np.array(data, dtype=float)

        # TODO: Multiple detection heads for handling N labels

        """
        Select the required label index
        0 - moving / stationary
        1 - category / class
        2 - object distance 
        """

        labels, bboxes, area = np.array([]), np.array([]), np.array([])

        if len(data) > 0:
            labels = data[:, 1].astype('int64')

            # Convert bounding box from yolo to cartesian
            bboxes = data[:, -4:]
            bboxes = np.where(bboxes < 0, 0, bboxes)

            # bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
            # bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
            # bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            # bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

            # bboxes[:, 0], bboxes[: 2] = 1920 * bboxes[:, 0], 1920 * bboxes[: 2]
            # bboxes[:, 1], bboxes[: 3] = 1080 * bboxes[:, 1], 1080 * bboxes[: 3]

            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        return labels, bboxes, area

    def __getitem__(self, idx):
        img_path = self.images_dir[idx]
        image = Image.open(img_path).convert("RGB")
        # height, width = image.size

        labels, bboxes, area = SMDDataset.read_yolo(self.labels_dir[idx])

        target = dict()
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(bboxes)), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


if __name__ == '__main__':

    # Extract frames from SMD videos
    # smd_to_yolo("NIR")
    # smd_to_yolo("VIS_Onboard")
    # smd_to_yolo("VIS_Onshore")

    # Load dataset and test one image
    dataset = SMDDataset('../SMD/NIR/data_yolo/')
    print(dataset[0])
