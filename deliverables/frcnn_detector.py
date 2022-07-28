
import torch
import numpy as np

from frcnn import get_frcnn_inference


class iClass:
    def __init__(self):
        self.model = None
        self.half = False
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.imgsz = 640

    def initialize(self, model_path, device):
        # Initialize
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        print(self.device)
        self.model = get_frcnn_inference(model_path=model_path)

        # TODO: check this
        # self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size

        if self.half:
            self.model.half()  # to FP16

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        print("run once DONE")

    # print("Load model done")

    def detect(self, imgInput, conf, iou_thres):
        imgInput = np.moveaxis(imgInput, -1, 0)
        pred_output = []
        # print("Run inference")
        img = torch.from_numpy(imgInput).to(self.device)
        # print(img.size())
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # print(img.size())
        # print("Inference")
        pred = self.model(img)[0]
        # print("Inference done")

        # Apply NMS
        # TODO: Check this too
        # pred = non_max_suppression(pred, conf, iou_thres, classes=None, agnostic=False)
        # print(pred)
        # print("non_max_suppression done")

        # # Process detections
        # for i, det in enumerate(pred):  # detections per image
        #     if len(det):
        #         for *xyxy, conf, cls in reversed(det):
        #             pred_output.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf.item(), cls.item()])

        # print(pred_output)

        # TODO: process pred -> pred_output
        return pred_output
