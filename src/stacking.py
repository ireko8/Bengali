import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

from dataset import val_split
from utils import setup, count_parameter, get_lr, load_csv, now
from loss import LabelSmoothedCE

from train_val_predict import train, validate, predict
from augment import train_transform, valid_transform
from models.resnet import ResNet

from config import conf


def main():

    exp_name = f'baseline_{now()}'
    device, log, result_dir = setup(exp_name, conf)

    train_df = load_csv(conf.train_csv)
    if conf.npy:
        train_images = np.load(conf.train_images)
    else:
        train_images = pd.read_parquet(conf.train_images)

    train_df["gr"] = 0
    train_df["cd"] = 0
    train_df["vd"] = 0
    train_df["image_mean"] = 0

    models = [f"se_resnext50_f{i}.pkl" for i in range(5)]

    preds = np.zeros((len(train_df), conf.gr_size + conf.vd_size + conf.cd_size))
    image_stats = np.zeros((len(train_df), 2))

    log.info('done')
    for i in range(5):

        model = ResNet(conf, arch_name=conf.arch,
                          input_size=conf.image_size)
        model.load_state_dict(torch.load(models[i]))
        model.to(device)

        ds = val_split(train_df, train_images, fold=i)
        _, val_ds, _, val_images = ds['train'], ds['val'], ds['train_images'], ds['val_images']

        test_preds = predict(model, val_ds, val_images, valid_transform,
                             device)

        print(test_preds.shape)
        te_ind = ds['te_ind']
        preds[te_ind] += test_preds
        image_stats[te_ind, 0] = val_images.mean((1, 2))
        image_stats[te_ind, 0] = val_images.std((1, 2))

    preds = np.concatenate([preds, image_stats], axis=1)

    for t in ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]:
        rf = RandomForestClassifier(n_jobs=16)
        # train = xgb.DMatrix(preds, label=train_df[t])
        # params = {"max_depth": 4, "nthread": 16, "objective": "multi:softmax",
        #           "eval_metric": ["merror", "mlogloss"], "num_class": conf.gr_size}
        # xgb.cv(params, train, num_boost_round=1000, nfold=5, seed=conf.seed,
        #        early_stopping_rounds=40, verbose_eval=10)
        rf.fit(preds, train_df[t])
        with open(f"{t}_rf2.pkl", "wb") as f:
            joblib.dump(rf, f)


if __name__ == "__main__":
    main()
