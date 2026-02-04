from pathlib import Path
import numpy as np
from ruamel.yaml import YAML
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from model.train.dataset import SynthGenerator, XPSDataset


class XPSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        params_path="model/params.yaml",
        train_data=None,
        val_data=None,
        synth_data=None,
        train=None,
        seed=None,
        generate_synth=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        params = {}
        if params_path:
            yaml_loader = YAML(typ="safe", pure=True)
            params = yaml_loader.load(Path(params_path))

        self.seed = seed
        self.train_data = train_data
        self.val_data = val_data
        self.synth_data = params.get("synth_data")
        self.train_params = train if train is not None else {}

        self.generate_synth = generate_synth
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        if not self.generate_synth:
            return

        data_generator = SynthGenerator(self.synth_data, self.seed)
        data_generator.gen_dataset(self.train_data)

    def setup(self, stage=None):
        dataset = XPSDataset(self.train_data)

        has_real_val = self.has_real_val_data(self.val_data)
        if has_real_val:
            self.train_dataset = dataset
            self.val_dataset = XPSDataset(self.val_data)
            print(
                f"Using real validation data: train={len(self.train_dataset)} "
                f"val={len(self.val_dataset)}"
            )
        else:
            split = self.train_params.get("train_test_split", 0.8)
            generator = self.seeded_generator(self.seed)
            train_size = int(len(dataset) * split)
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, (train_size, val_size), generator=generator)
            
            print(
                f"Using split validation: train={len(self.train_dataset)} "
                f"val={len(self.val_dataset)} (split={split})")

    def train_dataloader(self):
        batch_size = self.train_params.get("batch_size", 128)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        batch_size = self.train_params.get("batch_size", 128)
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def seeded_generator(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return torch.Generator().manual_seed(seed)

    def has_real_val_data(self, val_data_dir):
        if not val_data_dir:
            return False
        val_path = Path(val_data_dir)
        return val_path.exists() and any(val_path.glob("*.csv"))
