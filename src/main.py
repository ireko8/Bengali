import copy
from time import time

import torch
from torch import nn, optim
import numpy as np
import pandas as pd

from dataset import val_split
from utils import setup, count_parameter, get_lr, load_csv, now
from loss import ReducedFocalLoss
from train_val_predict import train, validate, predict
from augment import train_transform, valid_transform
from models.resnet import ResNet
from models.attention_resnet import AttentionResNet
from models.densenet import DenseNet
from models.efficientnet import EfficientNet
from config import conf


def train_model(train_df,
                train_images,
                test_df,
                test_images,
                base_model,
                criterion,
                log,
                device,
                exp_dir,
                fold=0,
                num_epoch=1,
                mask_epoch=1):

    ds = val_split(train_df, train_images, fold=fold)
    learn_start = time()

    log.info('classification learning start')
    log.info("-" * 20)
    model = base_model.to(device)
    # log.info(model)
    log.info(f'parameters {count_parameter(model)}')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_recall = 0

    # Observe that all parameters are being optimized
    log.info('Optimizer: Adam')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=conf.init_lr)  #, weight_decay=1e-5)

    log.info(
        f"Scheduler: CosineLR, period={conf.period}")
    train_ds, val_ds, train_images, val_images = ds['train'], ds['val'], ds['train_images'], ds['val_images']

    # scheduler = optim.lr_scheduler.CyclicLR(
    #     optimizer, conf.eta_min, conf.init_lr, cycle_momentum=False,
    #     step_size_up=300,
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max',
        patience=20, threshold=0.001,
        threshold_mode="abs",
    )
    
    for epoch in range(num_epoch):
        try:
            start = time()

            _, train_res = train(model, optimizer, # scheduler, 
                                 train_ds, train_images,
                                 train_transform,
                                 device, criterion, epoch=epoch)

            clf_loss = train_res['loss']
            val_preds, val_res = validate(model, val_ds, val_images,
                                          valid_transform,
                                          device, criterion)
            val_clf = val_res['loss']
            val_recall = val_res['recall']

            calc_time = time() - start
            accum_time = time() - learn_start
            lr = get_lr(optimizer)

            log_msg = f"{epoch}\t{calc_time:.2f}\t{accum_time:.1f}\t{lr:.4f}\t"            
            log_msg += f"{clf_loss:.4f}\t"

            train_recall = train_res['recall']
            log_msg += f"{train_recall:.4f}\t"
            log_msg += f"{val_clf:.4f}\t{val_recall:.4f}\t"
            log.info(log_msg)
            scheduler.step(val_recall)

            if val_recall > best_recall:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_recall = val_recall
                best_val_preds = val_preds
                torch.save(model.state_dict(), exp_dir/f'model_{fold}.pkl')
                np.save(exp_dir/f'val_preds_{fold}.npy', val_preds)

        except KeyboardInterrupt:
            break

    log.info("-" * 20)
    log.info('Best val Recall: {:4f}'.format(best_recall))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # test_preds = predict(model, test_df, test_images, valid_transform,
    #                      device)

    return model, best_val_preds  # , test_preds


def main():

    exp_name = f'baseline_{now()}'
    device, log, result_dir = setup(exp_name, conf)

    train_df = load_csv(conf.train_csv)
    if conf.npy:
        train_images = np.load(conf.train_images)
    else:
        train_images = pd.read_parquet(conf.train_images)

    test_df = load_csv(conf.test_csv)
    if conf.npy:
        test_images = np.load(conf.test_images)
    else:
        test_images = pd.read_parquet(conf.test_images)

    log.info('done')
    for i in range(5):
        if i != conf.fold:
            continue

        if "resnet" in conf.arch or "resnext" in conf.arch:
            model_ft = ResNet(conf, arch_name=conf.arch,
                              input_size=conf.image_size)
        elif "densenet" in conf.arch:
            model_ft = DenseNet(conf, arch_name=conf.arch,
                                input_size=conf.image_size)
        elif "efficientnet" in conf.arch:
            model_ft = EfficientNet(conf, arch_name=conf.arch)

        criterion = [
            nn.CrossEntropyLoss(reduction="none"),
            nn.CrossEntropyLoss(reduction="none"),
            nn.CrossEntropyLoss(reduction="none")
        ]
        criterion = [c.to(device) for c in criterion]

        model_ft, val_preds = train_model(
            train_df,
            train_images,
            test_df,
            test_images,
            model_ft,
            criterion,
            log,
            device,
            result_dir,
            fold=i,
            num_epoch=conf.num_epoch)

        torch.save(model_ft.state_dict(), result_dir/f'model_{i}.pkl')
        np.save(result_dir/f'val_preds_{i}.npy', val_preds)
        # np.save(result_dir/f'test_preds_{i}.npy', test_preds)


if __name__ == "__main__":
    main()
