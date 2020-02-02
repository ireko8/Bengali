import cv2
import numpy as np
from sklearn.model_selection import KFold
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
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample = dict()
        image, row_label = self.load_image(index)
        
        if conf.mask and self.test != "test":
            sample["mask"] = self.mask_target(image, conf.mask_size,
                                              row_label.grapheme_root)
        
        image = image[:, :, np.newaxis] / 255
        
        if self.augment:
            image = self.augment(image=image)['image']
            image = np.moveaxis(image, -1, 0)

        # image = (image - 0.06923) / 0.2052  # normalize
        image = (image - 0.05503) / 0.1722
        
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
    mskf = KFold(n_splits=int(1 / val_size), shuffle=True, random_state=conf.seed)
    splitter = mskf.split(df.index)
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
            class_count = df.diagnosis.value_counts()
            class_count = 1 / class_count
            df['weight'] = df.diagnosis.map(class_count)
            sampler = WeightedRandomSampler(df.weight, len(df))
    else:
        drop_last = False
    loader = DataLoader(
        ds, batch_size=batch_size, # shuffle=shuffle,
        num_workers=conf.num_workers,
        sampler=sampler,
        drop_last=drop_last)
    return loader, len(ds)
