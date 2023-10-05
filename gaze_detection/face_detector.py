import numpy as np
import torch


from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class FaceDetector:
    def __init__(self, device: str):
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load("weights/yolov7-tiny.pt", map_location=self.device)
        if self.half:
            self.model.half()  # to FP16

    def detect_face(self, file_path: str):
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size

        dataset = LoadImages(file_path, img_size=imgsz, stride=stride)
        for _, img_, im0s, _ in dataset:
            img = torch.from_numpy(img_).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0 
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.5, 0.45, agnostic=False)
            if len(pred) > 0:
                det = pred[0]
                im0 = im0s.copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    images = []
                    rects = []
                    for *xyxy, _, _ in reversed(det):
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        images.append(im0.astype(np.uint8)[
                            max(int(y1) - h // 4, 0):min(int(y2) + h // 4, im0.shape[0]),
                            max(int(x1) - w // 4, 0):min(int(x2) + w // 4, im0.shape[1]),
                        ])
                        rects.append([int(x1), int(y1), w, h])
                else:
                    return img_, None
            else:
                return img_, None
        return images[0], rects[0]