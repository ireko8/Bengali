from albumentations import Compose, OneOf, Normalize, ShiftScaleRotate
from albumentations import IAAAdditiveGaussianNoise, OpticalDistortion, GridDistortion
from albumentations import Cutout, Rotate

augmix_transform = [
    ShiftScaleRotate(rotate_limit=15, p=1),
    OpticalDistortion(p=1),
    Cutout(max_h_size=8, max_w_size=8, p=1),
    IAAAdditiveGaussianNoise(p=1),
    # GridDistortion(p=1),
    Rotate(limit=15, p=1)
]
    
train_transform = Compose([
    ShiftScaleRotate(rotate_limit=15, p=0.5),
    # IAAAdditiveGaussianNoise(p=0.3),
    OneOf([
            OpticalDistortion(p=1),
            GridDistortion(p=1),
        ], p=0.2),
    # Cutout(max_h_size=8, max_w_size=8, p=0.2),
])

valid_transform = Compose([
])