import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os
import sys
import numpy as np
import inspect
import torch
from pathlib import Path
from loguru import logger

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from facemesh import FaceMesh
from face_detector import FaceDetector
from warnings import filterwarnings

filterwarnings("ignore")

class EyesExtractor:
    def __init__(
        self,
        facemesh_weights: str = "./weights/facemesh.pth",
        device: str = "cpu"
    ):
        self.facemesh =  FaceMesh().to(device)
        self.facemesh.load_weights(facemesh_weights)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155,
                                 246, 161, 160, 159, 158, 157, 173, 133]
        self.right_eye_indices = [249, 263, 362, 373, 374, 380, 381, 382,
                                  384, 385, 386, 387, 388, 390, 398, 466]

    def get_eye_crop(self, points: np.array, img: np.array, which_eye: str):
        h, w, _ = img.shape
        sorted_point_x = sorted(points, key = lambda x: x[0])
        sorted_point_y = sorted(points, key = lambda x: x[1])
        left_point = sorted_point_x[0]
        right_point = sorted_point_x[-1]
        up_point = sorted_point_y[0]
        down_point = sorted_point_y[-1]
        multiplicator = 1/2
        margin_x_left = max(64 - (right_point[0] - left_point[0]), 0) * multiplicator
        margin_x_right = max(64 - (right_point[0] - left_point[0]), 0) * (1 - multiplicator)
        margin_y = max(64 - (down_point[1] - up_point[1]), 0) / 2
        min_x, max_x = max(0, int(left_point[0] - margin_x_left)), min(w, int(right_point[0] + margin_x_right))
        min_y, max_y = max(0, int(up_point[1] - margin_y)), min(h, int(down_point[1] + margin_y))
        img_eye = img[min_y: max_y, min_x: max_x, :]
        left_top_point = (min_x, min_y) 
        return img_eye, left_top_point


    def extract_eyes(self, face_crop: np.ndarray):
        face_crop = cv2.resize(face_crop, (192, 192), interpolation = cv2.INTER_LINEAR)
        face_crop = cv2.flip(face_crop, 1)
        facemesh_points = self.facemesh.predict_on_image(face_crop).cpu().numpy()
        left_eye_points = facemesh_points[self.left_eye_indices]
        right_eye_points = facemesh_points[self.right_eye_indices]
        img_eye_left, _ = self.get_eye_crop(left_eye_points, face_crop, which_eye = "left")
        img_eye_right, _ = self.get_eye_crop(right_eye_points, face_crop, which_eye = "right")
        img_eye_left = cv2.resize(img_eye_left, (64, 64), interpolation = cv2.INTER_LINEAR)
        img_eye_right = cv2.resize(img_eye_right, (64, 64), interpolation = cv2.INTER_LINEAR)
        return img_eye_left, img_eye_right


class GazeDetectionDataset(Dataset):
    """Gaze detection dataset."""

    def __init__(
        self,
        data: pd.DataFrame,
        transform_list: Optional[List]= None,
        transform: Optional[A.Compose]= None,
        to_tensors: bool = False,
        device: str = "cpu",
        inference: bool = False,
        screen_features: bool = False,
        precompute_folder: str = "./data_precompute",
        augmentation_factor: int = 3,
    ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = data.copy()
        self.to_tensors = to_tensors
        transform_list_ = transform_list.copy()
        if to_tensors:
            transform_list_.append(ToTensorV2())
        transform_list_eyes = []
        if to_tensors:
            transform_list_eyes.append(ToTensorV2())
        self.transforms = A.Compose(transform_list_)
        self.transform_eyes = A.Compose(transform_list_eyes)
        transform_list_mask = [A.Resize(25, 25)]
        if to_tensors:
            transform_list_mask.append(ToTensorV2())
        self.transform_mask = A.Compose(transform_list_mask)
        self.device = torch.device(device)
        self.faceCascade = cv2.CascadeClassifier("./weights/haarcascade_frontalface_alt.xml")
        self.eyesExtractor = EyesExtractor(device = device)
        self.inference = inference
        if device != "cpu":
            device = str(0)
        self.face_detector = FaceDetector(device = device)
        self.screen_features = screen_features
        self.precompute_folder = precompute_folder
        if not os.path.exists(precompute_folder):
            os.mkdir(precompute_folder)
            os.mkdir(os.path.join(precompute_folder, "faces"))
            os.mkdir(os.path.join(precompute_folder, "eyes"))
            os.mkdir(os.path.join(precompute_folder, "bboxes"))
        ## Augmentations
        self.transform = transform
        self.augmentation_factor = augmentation_factor


    def get_face_mask(self, frame, face_rect):
        mask = np.zeros((frame.shape[0], frame.shape[1]))
        if face_rect is None:
            return mask
        mask[
            face_rect[1] : face_rect[1] + face_rect[3],
            face_rect[0] : face_rect[0] + face_rect[2]
        ] = 1
        return mask

    def __len__(self):
        return len(self.df) * self.augmentation_factor if self.transform is not None else len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx % self.df.shape[0]]
        img_path = row['paths']
        try:
            image_orig = cv2.imread(img_path)
            image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
            image, face_rect = self.face_detector.detect_face(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            face_mask = self.get_face_mask(image_orig, face_rect)
            image_eye_l, image_eye_r = self.eyesExtractor.extract_eyes(image)
            image = image / 255.
            image_eye_l = image_eye_l / 255.
            image_eye_r = image_eye_r / 255.
        except Exception:
            raise Exception(f"Image {img_path} failed loading")
        if not self.inference:
            coordinates = row[['x_normalized', 'y_normalized']].values.astype(np.float32)
        if self.screen_features:
            screen_feat = row[[
                'Screen Width (cm)',
                'Screen Height (cm)',
                'Distance From Screen (cm)'
            ]].values.astype(np.float32)
            screen_feat = (screen_feat - screen_feat.mean()) / screen_feat.std()
        
        if self.transforms:
            image = self.transforms(image = image)['image']
            face_mask = self.transform_mask(image = face_mask)['image']
            image_eye_l = self.transform_eyes(image = image_eye_l)['image']
            image_eye_r = self.transform_eyes(image = image_eye_r)['image']
            if self.to_tensors:
                if not self.inference:
                    coordinates = torch.from_numpy(coordinates).to(self.device)
                if self.screen_features:
                    screen_feat = torch.from_numpy(screen_feat).to(self.device)
                image = image.type(torch.float32).to(self.device)
                image_eye_l = image_eye_l.type(torch.float32).to(self.device)
                image_eye_r = image_eye_r.type(torch.float32).to(self.device)
                face_mask = face_mask.type(torch.float32).to(self.device)
            
        sample = {'image': image, "eye_l": image_eye_l,
                  "eye_r": image_eye_r, "face_mask": face_mask}
        if not self.inference:
            sample['coordinates'] = coordinates
        if self.screen_features:
            sample['screen_features'] = screen_feat
        return sample


if __name__ == '__main__':
    logger.info("Testing dataloader")
    frames_folder = "./real_experiment/frames_test/"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = Path(frames_folder).glob('*.png')
    paths = [str(path.absolute()) for path in p]
    df_files = pd.DataFrame({"paths": paths})
    df_files["ind"] = df_files.paths.apply(lambda x: float(Path(x).stem))
    df_files = df_files.sort_values("ind").set_index("ind")
    trans_list = [A.Resize(192, 192)]
    face_dataset = GazeDetectionDataset(data = df_files, transform_list=trans_list,
                                    to_tensors=True, device=device, inference=True, screen_features=False)
    print("img_size\t\t\teye_size\t\t\tmask_size")
    for i, sample in enumerate(face_dataset):
        print(i, sample['image'].size(), sample['eye_l'].size(), sample['face_mask'].size())
        if i == 3:
            break