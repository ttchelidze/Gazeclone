import numpy as np
import torch
import os
import torch.nn as nn

from warnings import filterwarnings, warn
from tqdm import tqdm
import sys
import inspect
from loguru import logger

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from facemesh import FaceMeshBlock, FaceMesh
from pupil_detection import IrisLM, IrisBlock

filterwarnings('ignore')
tqdm.pandas()


class FaceGridModel(nn.Module):
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EyesModel(nn.Module):
    def __init__(self, pretrained_model_eyes: nn.Module):
        super(EyesModel, self).__init__()
        self.backbone = pretrained_model_eyes.backbone
        self.regression_head_eyes = nn.Sequential(
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
        )
        # connect eyes
        self.fc = nn.Sequential(
            nn.Linear(2 * 128 * 1 * 1, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_eye_l, x_eye_r):
        x_eye_l = self.backbone(x_eye_l)
        x_eye_l = self.regression_head_eyes(x_eye_l)
        x_eye_l = x_eye_l.view(-1, 128 * 1 * 1)

        x_eye_r = self.backbone(x_eye_r)
        x_eye_r = self.regression_head_eyes(x_eye_r)
        x_eye_r = x_eye_r.view(-1, 128 * 1 * 1)
        x = torch.cat([x_eye_l, x_eye_r], 1)
        x = self.fc(x)
        return x


class FaceModel(nn.Module):
    def __init__(self, pretrained_model_face: nn.Module):
        super(FaceModel, self).__init__()
        self.backbone = pretrained_model_face.backbone
        self.regression_head_face = nn.Sequential(
            FaceMeshBlock(128, 128, stride=2),
            FaceMeshBlock(128, 128),
            FaceMeshBlock(128, 128),
            nn.Conv2d(128, 32, 1),
            nn.PReLU(32),
            FaceMeshBlock(32, 32),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def _preprocess(self, x):
        return x.to(torch.float32) * 2.0 - 1.0

    def forward(self, x_face):
        x_face = self._preprocess(x_face)
        x_face = nn.ReflectionPad2d((1, 0, 1, 0))(x_face)
        x_face = self.backbone(x_face)
        x_face = self.regression_head_face(x_face)
        x_face = x_face.view(-1, 32 * 3 * 3)
        x = self.fc(x_face)
        return x


class FineTuneModel(nn.Module):
    def __init__(
        self,
        pretrained_model_face: nn.Module,
        pretrained_model_eyes: nn.Module,
        screen_features: bool = False,
    ):
        super(FineTuneModel, self).__init__()
        self.face_model = FaceModel(pretrained_model_face)
        self.eyes_model = EyesModel(pretrained_model_eyes)
        self.face_grid_model = FaceGridModel()
        self.screen_features = screen_features
        if not screen_features:
            self.fc = nn.Sequential(
                nn.Linear(128+64+128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(128+64+128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 13),
                nn.ReLU(inplace=True),
            )
            self.layer_norm = nn.LayerNorm(16)
            self.fc2 = nn.Linear(16, 2)
            

    def _preprocess(self, x):
        return x.to(torch.float32) * 2.0 - 1.0
        
    def forward(self, x_face, x_eye_l, x_eye_r, x_grid, x_screen = None):
        if self.screen_features and x_screen is None:
            raise Exception("You should pass screen features")
        if not self.screen_features and x_screen is not None:
            warn("Screen fearures won't be used")
        x_eyes = self.eyes_model(x_eye_l, x_eye_r)
        x_face = self.face_model(x_face)
        x_grid = self.face_grid_model(x_grid)
        x = torch.cat([x_eyes, x_face, x_grid], axis = 1)
        if not self.screen_features:
            x = self.fc(x)
        else:
            x = self.fc1(x)
            x = torch.cat([x, x_screen], axis = 1)
            x = self.layer_norm(x)
            x = self.fc2(x)
        return x


if __name__ == '__main__':
    logger.info("Testing model on sample input of size 2")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rand_img = np.random.randint(0, 255, (2, 3, 192,192))
    x_face = torch.from_numpy(rand_img).to(device)
    rand_img = (np.random.randint(0, 255, (2, 3, 64, 64)) / 255.)
    x_eye = torch.from_numpy(rand_img).type(torch.float32).to(device)
    rand_grid = (np.random.randint(0, 255, (2, 25, 25)) / 255.)
    x_grid = torch.from_numpy(rand_grid).type(torch.float32).to(device)

    pretrained_model_face = FaceMesh()
    pretrained_model_face.load_weights("./weights/facemesh.pth")

    model_path = "./weights/irislandmarks.pth"
    pretrained_model_eyes = IrisLM()
    weights = torch.load(model_path)
    pretrained_model_eyes.load_state_dict(weights)
    model = FineTuneModel(pretrained_model_face, pretrained_model_eyes, screen_features=False).to(device)
    print("Result:")
    print(model(x_face, x_eye, x_eye, x_grid))