
import torch
import numpy as np

from retinanet import get_retina_model


class iClass:
    def __init__(self):
        self.model = None
        self.half = False
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.imgsz = 640

    def initialize(self, model_path=None, device=None):
        # Initialize
        # Half precision affecting predictions
        # self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if not model_path:
            print("Missing model checkpoint")
        self.model = get_retina_model(model_path=model_path)
        self.model.eval()

        # TODO: image sizes not needed?
        # self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size

        if self.half:
            self.model.half()  # to FP16

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        print("run once DONE")

    def detect(self, imgInput, conf=0.7, iou_thres=None):
        # imgInput -> numpy/ cv2 image
        imgInput = np.moveaxis(imgInput, -1, 0)
        pred_output = []
        # print("Run inference")
        img = torch.from_numpy(imgInput).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            for i in range(len(det['boxes'])):
                score = det['scores'][i].item()
                if score > conf:
                    xyxy = det['boxes'][i].detach().cpu().numpy()
                    cls = det['labels'][i].item()
                    pred_output.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], cls, score])

        print(len(pred_output), pred_output)
        # TODO: process pred -> pred_output
        return pred_output


if __name__ == '__main__':
    # Test snippet
    model = iClass()
    model.initialize("models/Retina-1.pth")

    import cv2
    img = cv2.imread("../test.jpg")
    model.detect(img)
