from pathlib import Path
from time import time
from datetime import datetime

import numpy as np

import torch


class Trainer():
    """Class for model training and evaluation."""
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            metric_functions={},
            # path_to_model='model/trained_models/model.pt',
            log_dir=None,
            device=torch.device('cpu')
    ):
        """
        Initialize the Trainer class.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be trained.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        criterion : torch.nn.Module
            Loss function for training.
        optimizer : torch.optim.Optimizer
            Optimizer for the model.
        metric_functions : dict, optional
            Dictionary of metric functions to be used for evaluation. The keys are the names of the metrics
            and the values are the functions to calculate them.
        path_to_model : str, optional
            Path to save the trained model. The default is 'model/trained_models/model.pt'.
        log_dir : str, optional
            Path to save the log of the training. The default is 'train_log_<datetime>.txt'.
        device : torch.device, optional
            Device to use for training. The default is torch.device('cpu').
        """

        model.to(device)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        if metric_functions:
            self.metric_functions = metric_functions

        # if path_to_model:
        #     self.path_to_model = Path(path_to_model)
        # else:
        #     self.path_to_model = Path('model/trained_models/model.pt')


        if log_dir:
            p = Path(log_dir)
            p.mkdir(exist_ok=True)
            self.log_dir = p
        else:
            p = Path(f'train_log_{datetime.now().strftime("%Y%m%d_%H%M")}')
            p.mkdir(exist_ok=True)
            self.log_dir = p
        
        self.path_to_model = p.joinpath('model.pt')
        self.log = p.joinpath(f'log.csv')
        self.create_log()

    def train(self, epochs):
        
        best_val_loss = np.inf

        for epoch in range(epochs):
            ep_start = time()
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            ep_end = time()

            self.write_log(epoch, train_loss, val_loss, metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time per epoch: {ep_end - ep_start:.1f}s')
                self.save_checkpoint(epoch, best_model, train_loss, val_loss)

        print(f'Epoch {epoch+1}, train loss: {train_loss:.4f}, best val loss: {best_val_loss:.4f}')
        print('Train finished!')
        torch.save(best_model, self.path_to_model)
        print(f'Model weights saved to {self.path_to_model.absolute()}')
    
    def train_epoch(self):
        self.model.train()
        train_loss = 0.0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(self.train_loader)
        return train_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0

        metrics = {metric: 0.0 for metric in self.metric_functions.keys()}

        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            val_loss += loss.item()

            for metric in self.metric_functions.keys():
                metrics[metric] += self.metric_functions[metric](output, target).item()

        for metric in metrics:
            metrics[metric] /= len(self.val_loader)
        
        val_loss = val_loss / len(self.val_loader)
        return val_loss, metrics

    def save_checkpoint(self, epoch, state_dict, train_loss, val_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, self.log_dir.joinpath(f'check.pt'))
    
    def create_log(self):
        with open(self.log, 'w') as f:
            cols = ['epoch', 'train_loss', 'val_loss']
            cols.extend(self.metric_functions.keys())
            metrics_to_write = ','.join(cols)
            f.write(metrics_to_write + '\n')

    def write_log(self, epoch, train_loss, val_loss, metrics):
        vals = [epoch, train_loss, val_loss]
        vals.extend(metrics.values())
        metrics_to_write = ','.join([f'{v:.4f}' for v in vals])
        with open(self.log, 'a') as f:
            f.write(metrics_to_write + '\n')
