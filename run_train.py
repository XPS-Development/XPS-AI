import os
import random
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ruamel.yaml import YAML

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from tools import view_labeled_data
from model.train.dataset import SynthGenerator, XPSDataset
from model.train.metrics import IoU, Accuracy, Precision, Recall, DiceFocalLoss, IoULoss
from model.models.model_deeper import XPSModel
from model.train.trainer import Trainer
from tools import Analyzer, Spectrum


def load_params():
    yaml_loader = YAML(typ='safe', pure=True)
    params = yaml_loader.load(Path('model/params.yaml'))

    seed = params['seed']
    path_to_data = params['data_path']

    return seed, path_to_data, params['train'], params['synth_data']


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    return gen


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@torch.no_grad()
def test_model(test_dir, model, save_dir):
    model.eval()
    a = Analyzer(model)
    for f in test_dir.iterdir():
        if save_dir:
            save_path = save_dir.joinpath(f.with_suffix('.png').name)
        else:
            save_path = None
        
        array = np.loadtxt(f, delimiter=',', skiprows=3)
        x, y = array[:, 0], array[:, 1]
        spectrum = Spectrum(x, y)
        a.predict(spectrum)
        x, y = spectrum.get_data()
        view_labeled_data(x, y, spectrum.get_masks(), save_path=save_path)

def save_train_log_png(self, path: Path):
    with open(path, 'r') as f:
        log_data = pd.read_csv(f)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    metrics = log_data.columns.to_list()[3:]
    log_data.plot(x='epoch', y=['train_loss', 'val_loss'], kind='line', ax=axs[0])
    log_data.plot(x='epoch', y=metrics, kind='line', ax=axs[1])
    for a in axs:
        a.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False)

    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight') 


def main():
    
    seed, path_to_data, train_params, synth_params = load_params()
    gen = fix_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Generating data...')
    data_generator = SynthGenerator(synth_params, seed)
    data_generator.gen_dataset(path_to_data)
    dataset = XPSDataset(path_to_data)
    print(f'Dataset size: {len(dataset)}')

    split = train_params['train_test_split']
    train_data, val_data = random_split(dataset, (split, 1-split), gen)    


    train_dl = DataLoader(train_data, batch_size=train_params['batch_size'], shuffle=True, generator=gen)
    val_dl = DataLoader(val_data, batch_size=train_params['batch_size'], shuffle=False)
    model = XPSModel()
    optimizer = Adam(model.parameters(), lr=train_params['learning_rate'])
    dicefocal = DiceFocalLoss()
    iouloss = IoULoss()

    criterion = lambda inp, tar: iouloss(inp[:, 0, :], tar[:, 0, :]) + 5 * dicefocal(inp[:, 1, :], tar[:, 1, :])

    iou = IoU()
    acc = Accuracy()
    prec = Precision()
    rec = Recall()
    
    metrics = {
        'iou_peak': lambda inp, tar: iou(inp[:, 0, :], tar[:, 0, :]),
        'acc_peak': lambda inp, tar: acc(inp[:, 0, :], tar[:, 0, :]),
        'prec_peak': lambda inp, tar: prec(inp[:, 0, :], tar[:, 0, :]),
        'rec_peak': lambda inp, tar: rec(inp[:, 0, :], tar[:, 0, :]),
        'iou_max': lambda inp, tar: iou(inp[:, 1, :], tar[:, 1, :]),
        'acc_max': lambda inp, tar: acc(inp[:, 1, :], tar[:, 1, :]),
        'prec_max': lambda inp, tar: prec(inp[:, 1, :], tar[:, 1, :]),
        'rec_max': lambda inp, tar: rec(inp[:, 1, :], tar[:, 1, :])
    }
    
    print('Training model...')
    trainer = Trainer(model, train_dl, val_dl, optimizer, criterion, metrics, device=device)
    trainer.train(train_params['num_epochs'])
    # test_model(Path('other/test_data'), model, trainer.log_dir)

    # save_train_log_png(trainer.log)

if __name__ == '__main__':
    main()
