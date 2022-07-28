
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from torchsummary import summary


class iClass:
    def __init__(self):
        self.model = None
        self.half = False
        self.device = select_device('0')
        self.imgsz = 640

    def initialize(self, xi_weights, device):
        # Initialize
        # set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        print(xi_weights)
        print(self.device)
        # print("Load model")
        self.model = attempt_load(xi_weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # print(self.model)
        # summary(self.model, (3, 640, 640))
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        print("run once DONE")

    # print("Load model done")

    def detect(self, imgInput, conf, iou_thres):
        # print("access yolov5")
        imgInput = np.moveaxis(imgInput, -1, 0)
        # print(imgInput.shape, conf, iou_thres)
        pred_output = []
        # print("Run inference")
        img = torch.from_numpy(imgInput).to(self.device)
        # print(img.size())
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # print(img.size())
        # print("Inference")
        pred = self.model(img, augment=False)[0]
        # print("Inference done")

        # Apply NMS
        pred = non_max_suppression(pred, conf, iou_thres, classes=None, agnostic=False)
        # print(pred)
        # print("non_max_suppression done")

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy, conf.item(), cls.item())
                    pred_output.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf.item(), cls.item()])

        # print(pred_output)
        return pred_output
