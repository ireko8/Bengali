import cv2
import numpy as np
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import conf


class BengalDataset(Dataset):
    def __init__(self,
                 metadata,
                 images,
                 augment=None,
                 test=False,
                 mixup=False):
        super().__init__()
        self.metadata = metadata
        self.images = images
        self.augment = augment
        self.test = test
        self.mixup = mixup

    def load_image(self, index):
        row_label = self.metadata.iloc[index]

        if conf.npy:
            image = self.images[index]
        else:
            row_image = self.images.iloc[index]
            image = row_image.drop('image_id').values.reshape(137, 236)
            
        # image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        return image, row_label

    @staticmethod
    def mask_target(image, mask_size, label):
        mask_array = np.zeros((conf.gr_size, mask_size[0], mask_size[1]))
        mask = cv2.resize(image, mask_size) / 255
        mask = mask > 0.1
        mask_array[label.astype(bool)] = mask.T
        return mask_array.astype(np.float32)

    def normalize(self, image):
        return (image - 0.05503) / 0.1722

    def augmix(self, image, width=3, depth=1, alpha=1.):
        """Perform AugMix augmentations and compute mixture.
        Args:
        image: Raw input image as float32 np.ndarray of shape (h, w, c)
        severity: Severity of underlying augmentation operators (between 1 to 10).
        width: Width of augmentation chain
        depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
        alpha: Probability coefficient for Beta and Dirichlet distributions.
        Returns:
        mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([alpha] * width))
        m = np.float32(np.random.beta(alpha, alpha))

        mix = np.zeros_like(np.moveaxis(image, -1, 0))
        
        for i in range(width):
            image_aug = image.copy()
            depth = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augment)
                image_aug = op(image=image_aug)['image']
                
                if image_aug.shape[0] != 1:
                    image_aug = np.moveaxis(image_aug, -1, 0)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.normalize(image_aug)

        image = np.moveaxis(image, -1, 0)
        mixed = (1 - m) * self.normalize(image) + m * mix
        return mixed
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample = dict()
        image, row_label = self.load_image(index)
        
        if conf.mask and self.test == "train":
            sample["mask"] = self.mask_target(image, conf.mask_size,
                                              row_label.grapheme_root)
        
        image = image[:, :, np.newaxis] / 255
        
        if self.augment:
            if conf.augmix:
                sample["mix1"] = self.augmix(image)
                sample["mix2"] = self.augmix(image)
            else:
                images = [self.augment(image=image)['image']
                          for _ in range(3)]
                image = np.concatenate(images, axis=-1)

        # image = (image - 0.06923) / 0.2052  # normalize
        image = np.moveaxis(image, -1, 0)
        image = self.normalize(image)
        
        if self.test != "test":
            label = [
                row_label.grapheme_root,
                row_label.vowel_diacritic,
                row_label.consonant_diacritic
            ]
            sample['label'] = np.array(label)

        sample['data'] = image.astype(np.float32)
        return sample
    

def val_split(df, images, val_size=0.2, fold=0, seed=71):
    if conf.stratify == "random":
        mskf = KFold(n_splits=int(1 / val_size), shuffle=True,
                     random_state=conf.seed)
    elif conf.stratify == "multilabel":
        mskf = MultilabelStratifiedKFold(n_splits=int(1 / val_size),
                                         shuffle=True, random_state=conf.seed)
    else:
        raise NotImplementedError

    y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]]
    splitter = mskf.split(df.index, y)

    for _ in range(fold + 1):
        tr_ind, te_ind = next(splitter)
    train_df = df.iloc[tr_ind].reset_index(drop=True)
    val_df = df.iloc[te_ind].reset_index(drop=True)
    if conf.npy:
        train_images = images[tr_ind]
        val_images = images[te_ind]
    else:
        train_images = images.iloc[tr_ind].reset_index(drop=True)
        val_images = images.iloc[te_ind].reset_index(drop=True)
    return {'train': train_df, 'val': val_df, 'train_images': train_images, 'val_images': val_images}


def worker_init_fn(worker_id):                                                 
    np.random.seed(conf.seed + worker_id)


def make_loader(df,
                images,
                batch_size=conf.batch_size,
                shuffle=True,
                test="train",
                worker_init_fn=worker_init_fn,
                **kwargs):

    ds = BengalDataset(
        df,
        images,
        test=test,
        **kwargs)

    sampler = None
    if test == "train":
        drop_last = True
        if conf.weighted_sample:
            class_count = df.grapheme_root.value_counts()
            class_count = 1 / class_count
            df['weight'] = df.grapheme_root.map(class_count)
            sampler = WeightedRandomSampler(df.weight, len(df))
            shuffle=False
    else:
        drop_last = False
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=conf.num_workers,
        sampler=sampler,
        drop_last=drop_last)
    return loader, len(ds)
