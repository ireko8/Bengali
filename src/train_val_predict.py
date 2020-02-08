import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score

from dataset import make_loader
from config import conf


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(data, target):
    lam = np.random.beta(conf.beta, conf.beta)
    rand_index = torch.randperm(data.size(0)).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    # compute output

    return data, target_a, target_b, lam


def criterion_for_each_target(criterion):
    def new_criterion(pred, labels):
        loss = 0
        for i, px in enumerate(pred):
            loss_i = criterion[i](px, labels[:, i])  # / batch_size
            loss += loss_i
        return loss
    return new_criterion


def calc_loss(pred, labels, criterion):
    pred_len = len(labels)
    pred_probs = list()
    pred_class = list()
    loss = 0
    for i, px in enumerate(pred):
        pred_probs.append(px.softmax(dim=1).cpu().data.numpy() / pred_len)
        pred_class.append(px.argmax(dim=1).cpu().data.numpy())
        if isinstance(criterion, list):
            loss_i = criterion[i](px, labels[:, i])
        else:
            loss_i = criterion(px, labels[:, i]) 
        w = 2 if i == 0 else 1
        loss += loss_i * w
    return loss, pred_probs, np.stack(pred_class, axis=1)


def mixup_data(x, y, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    alpha = conf.alpha
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_loss(criterion, pred, y_a, y_b, lam):
    loss = 0
    for i in range(3):
        p_i = pred[i]
        one_hot = torch.zeros(p_i.size(), device=conf.device_name)
        y_ai = one_hot.scatter_(1, y_a[:, i].view(-1, 1), 1)
        one_hot = torch.zeros(p_i.size(), device=conf.device_name)
        y_bi = one_hot.scatter_(1, y_b[:, i].view(-1, 1), 1)
        y_new = lam * y_ai + lam * y_bi
        loss += criterion[i](p_i, y_new)
    return loss


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    criterion = criterion_for_each_target(criterion)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def augmix_loss(orig_loss, pred_orig, pred_m1, pred_m2):
    
    loss = orig_loss
    for i in range(3):
        p_orig = pred_orig[i].softmax(dim=1)
        p_m1 = pred_m1[i].softmax(dim=1)
        p_m2 = pred_m2[i].softmax(dim=1)
    
        p_mixture = torch.clamp((p_orig + p_m1 + p_m2) / 3., 1e-7, 1).log()
        loss += (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                 F.kl_div(p_mixture, p_m1, reduction='batchmean') +
                 F.kl_div(p_mixture, p_m2, reduction='batchmean')) / 3
    return loss


def weighted_macro_recall(trues, preds):
    scores = list()
    accs = list()
    for i in range(3):
        s = recall_score(trues[:, i], preds[:, i], average='macro')
        scores.append(s)
        accs.append(accuracy_score(trues[:, i], preds[:, i]))
    return np.average(scores, weights=[2, 1, 1]), accs


def train(model,
          optimizer,
          # scheduler,
          train_df,
          train_images,
          aug,
          device,
          criterion,
          undersampling=False):

    model.train()
    dataloader, ds_size = make_loader(
        train_df,
        train_images,
        shuffle=True,
        test="train",
        mixup=conf.mixup,
        augment=aug)

    running_loss = 0.0
    all_trues = list()
    all_preds = list()

    # Iterate over data.
    pbar = tqdm(dataloader)
    optimizer.zero_grad()
    for i, sample in enumerate(pbar):
        inputs = sample['data'].to(device)
        labels = sample['label'].to(device)
        trues = labels.cpu().data.numpy()
        all_trues.append(trues)

        rand = np.random.random()
        if conf.augmix and rand < conf.augmix_prob:
            mix1 = sample['mix1'].to(device)
            mix2 = sample['mix2'].to(device)
            outputs = model(inputs)
            mix1 = model(inputs)
            mix2 = model(inputs)
            loss, _, pred_class = calc_loss(mix1, labels, criterion)
            loss = augmix_loss(loss, outputs, mix1, mix2)            
        elif conf.mixup and rand < conf.mixup_prob:
            inputs, ta, tb, lam = cutmix_data(inputs, labels)
            outputs = model(inputs)
            loss_a, _, pred_class = calc_loss(outputs, ta, criterion)
            loss_b, _, _ = calc_loss(outputs, tb, criterion)
            loss = lam * loss_a + (1-lam) * loss_b
        else:
            outputs = model(inputs)            
            loss, _, pred_class = calc_loss(outputs, labels, criterion)

        all_preds.append(pred_class)
        loss.backward()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        kaggle_score, each_scores = weighted_macro_recall(trues, pred_class)
        pbar.set_postfix({
            "loss": loss.item(),
            "kaggle": kaggle_score,
            "graphen": each_scores[0],
            "vowel": each_scores[1],
            "consonant": each_scores[2]
        })
        
        if (i + 1) % conf.accum_time == 0:
            optimizer.step()            
            optimizer.zero_grad()
            # scheduler.step()

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    epoch_loss = running_loss / ds_size
    epoch_kaggle, epoch_each = weighted_macro_recall(all_trues, all_preds)

    result = {
        'loss': epoch_loss,
        'recall': epoch_kaggle,
        "graphen": epoch_each[0],
        "vowel": epoch_each[1],
        "consonant": epoch_each[2]
    }
    return all_preds, result


def validate(model, val_df, val_images,
             aug,
             device,
             criterion):

    model.eval()

    dataloader, ds_size = make_loader(
        val_df,
        val_images,
        conf.batch_size,
        shuffle=False,
        test="valid",
        augment=aug)

    all_preds = []
    all_trues = []

    running_loss = 0.0

    # Iterate over data.
    for i, samples in enumerate(dataloader):
        with torch.set_grad_enabled(False):
            inputs = samples['data'].to(device)
            outputs = model(inputs)
            labels = samples['label'].to(device)

            outputs = model(inputs)            
            loss, _, pred_class = calc_loss(outputs, labels, criterion)
            all_preds.append(pred_class)            
            all_trues.append(labels.cpu().data.numpy())
            running_loss += loss.item() * inputs.size(0)

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    epoch_loss = running_loss / ds_size
    epoch_kaggle, epoch_each = weighted_macro_recall(all_trues, all_preds)

    result = {
        'loss': epoch_loss,
        'recall': epoch_kaggle,
        "graphen": epoch_each[0],
        "vowel": epoch_each[1],
        "consonant": epoch_each[2]
    }

    return all_preds, result


def predict(model, test_df,
            test_images,
            aug,
            device,
            data_dir='input/train'):

    model.eval()

    dataloader, ds_size = make_loader(
        test_df,
        test_images,
        conf.batch_size,
        shuffle=False,
        test="test",
        augment=aug)

    all_preds = []
    # Iterate over data.
    t = dataloader
    for i, samples in enumerate(t):
        with torch.set_grad_enabled(False):
            inputs = samples['data'].to(device)
            outputs = model(inputs)
            for px in outputs:
                all_preds.append(px.argmax(dim=1).cpu().data.numpy())

    all_preds = np.concatenate(all_preds)
    return all_preds
