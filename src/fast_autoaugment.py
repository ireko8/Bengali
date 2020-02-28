import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from albumentations import Cutout, ShiftScaleRotate, Compose, GridDistortion

from train_val_predict import validate
from config import conf
from models.resnet import ResNet


class FastAutoAugment:
    
    def __init__(self, val_df, val_images, model, depth=2):

        self.val = val_df
        self.val_images = val_images
        self.model = model

        self.depth = depth
        self.base_augment = [
            "Cutout",
            "ShiftScaleRotate"
        ]

    def policy_space(self, trial):

        augments = list()

        # Shiftscalerotate
        shift_limit = trial.suggest_uniform(f"shift_limit", 0, 0.25)
        scale_limit = trial.suggest_uniform(f"scale_limit", 0, 0.25)
        rotate_limit = trial.suggest_int(f"rotate_limit", 0, 45)
        p = trial.suggest_uniform(f"shiftscalerotate_prob", 0, 1)
        augments.append(
            ShiftScaleRotate(shift_limit=shift_limit,
                             scale_limit=scale_limit,
                             rotate_limit=rotate_limit, p=p))

        # GridDistortion
        num_steps = trial.suggest_int("num_steps", 1, 7)
        distort_limit = trial.suggest_uniform("distort_limit", 0, 0.5)
        p = trial.suggest_uniform(f"griddistortion_prob", 0, 1)
        augments.append(
            GridDistortion(num_steps=num_steps, distort_limit=distort_limit, p=p))
        
        # Cutout
        n_hole = trial.suggest_int("num_holes", 1, 16)
        h_size = trial.suggest_int("max_h_size", 1, 20)
        w_size = trial.suggest_int("max_w_size", 1, 20)
        p = trial.suggest_uniform(f"cutout_prob", 0, 1)
        augments.append(
            Cutout(num_holes=n_hole, max_h_size=h_size, max_w_size=w_size, p=p))
        
        return Compose(augments)

    def test(self):
        device = torch.device(conf.device_name)
        self.model.to(device)
        criterion = [
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss()
        ]
        criterion = [c.to(device) for c in criterion]

        all_preds, result = validate(
            self.model,
            self.val,
            self.val_images,
            Compose([]),
            device,
            criterion,
        )        
        print(result)

    def search(self, trial):

        augment = self.policy_space(trial)
        device = torch.device(conf.device_name)
        self.model.to(device)
        criterion = [
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss()
        ]
        criterion = [c.to(device) for c in criterion]

        all_preds, result = validate(
            self.model,
            self.val,
            self.val_images,
            augment,
            device,
            criterion,
        )        

        print(result)
        return result["loss"]


def search_policy():

    model = ResNet(conf, arch_name=conf.arch,
                   input_size=conf.image_size)
    model.load_state_dict(
        torch.load("result/baseline_2020_02_20_14_35_56/model_0.pkl")
    )

    val_df = pd.read_csv("val_index.csv")
    val_images = np.load("val_images.npy")

    faa = FastAutoAugment(val_df, val_images, model)

    # faa.test()
    study = optuna.create_study()
    study.optimize(faa.search, n_trials=200)

    print(study.best_trial)
    print(study.best_params)


if __name__ == "__main__":
    search_policy()
