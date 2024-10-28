import os
import random
from pathlib import Path
from time import time

import numpy as np
from matplotlib import pyplot as plt
from dvclive import Live
from ruamel.yaml import YAML

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from model.train.metrics import IoU, Accuracy, Precision, Recall, DiceFocalLoss, IoULoss
from model.train.dataset import XPSDataset
from model.model_deeper import XPSModel


def load_params():
    yaml_loader = YAML(typ='safe', pure=True)
    params = yaml_loader.load(Path('model/params.yaml'))

    seed = params['seed']
    path_to_data = params['data_path']

    return seed, path_to_data, params['train']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@torch.no_grad()
def evaluate(model, val_loader, criterion, metrics=()):

    results = {}
    val_loss = 0
    results.update([(f'peak_{name}', 0) for name, f in metrics])
    results.update([(f'max_{name}', 0) for name, f in metrics])

    l = len(val_loader)
    for x, peak_mask, max_mask in val_loader:
        pred_peak_mask, pred_max_mask = model(x)
        peak_targets = peak_mask.view(-1)
        max_targets = max_mask.view(-1)

        peak_inputs = pred_peak_mask.view(-1)
        max_inputs = pred_max_mask.view(-1)

        for name, func in metrics:
            results[f'peak_{name}'] += func(peak_inputs, peak_targets).detach().numpy() / l
            results[f'max_{name}'] += func(max_inputs, max_targets).detach().numpy() / l

        val_loss += criterion(peak_inputs, peak_targets, max_inputs, max_targets).detach().numpy() / l

    return val_loss, results

def train_one_epoch(model, train_loader, optimizer, criterion):
    sum_train_loss = 0

    for x, peak_mask, max_mask in train_loader:
        pred_peak_mask, pred_max_mask = model(x)

        train_loss = criterion(pred_peak_mask, peak_mask, pred_max_mask, max_mask)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        sum_train_loss += train_loss.detach().numpy()

    return sum_train_loss / len(train_loader)


if __name__ == '__main__':
    # init train params
    seed, path_to_data, params = load_params()
    num_epoch = params['num_epoch']
    batch_size = params['batch_size']
    split = params['train_test_split']
    lr = params['learning_rate']

    # fixing random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    # load model and data
    model = XPSModel()
    data = XPSDataset(path_to_data)

    train_data, val_data = random_split(data, (split, 1-split), gen)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        worker_init_fn=seed_worker,
        generator=gen
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        worker_init_fn=seed_worker,
        generator=gen
    )

    optimizer = Adam(model.parameters(), lr=lr)
    dicefocal = DiceFocalLoss()
    iouloss = IoULoss()
    criterion = lambda p_inp, p_tar, m_inp, m_tar: iouloss(p_inp, p_tar) + 5 * dicefocal(m_inp, m_tar)
    metrics = (
        ('accuracy', Accuracy()),
        ('precision', Precision()),
        ('recall', Recall()),
        ('iou', IoU())
    )

    with Live() as live:
        path = 'model/train/trained_models/model.pth'

        for epoch in range(num_epoch):
            best_val_loss = np.inf

            s = time()
            model.train()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            model.eval()
            val_loss, out_metrics = evaluate(model, val_loader, criterion, metrics)
            f = time()

            if val_loss < best_val_loss:
                best_state = model.state_dict()
                best_val_loss = val_loss

            live.log_metric('train_loss', train_loss, timestamp=True)
            live.log_metric('val_loss', val_loss)
            for metric_name, val in out_metrics.items():
                live.log_metric(metric_name, val)
            live.next_step()
            print(f'Epoch {epoch} finished at {f-s:.1f}s')
        
        model.load_state_dict(best_state)
        torch.save(model, path)
