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
        generate_synth_train=True,
        generate_synth_val=True,
        synth_train_size=None,
        synth_val_size=None,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    ) -> None:
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
        self.generate_synth_train = generate_synth_train
        self.generate_synth_val = generate_synth_val
        self.synth_train_size = synth_train_size
        self.synth_val_size = synth_val_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        if not self.generate_synth:
            return

        data_generator = SynthGenerator(self.synth_data, self.seed)
        train_path = Path(self.train_data)
        val_path = Path(self.val_data)

        train_size = int(self.synth_train_size or self.synth_data.get("dataset_size", 0))
        if train_size <= 0:
            train_size = 1
        val_size = int(self.synth_val_size or max(1, train_size // 4))

        real_train_count = self.csv_count(train_path)
        real_val_count = self.csv_count(val_path)

        added_train = 0
        added_val = 0

        if self.generate_synth_train:
            train_start = self.next_csv_index(train_path)
            data_generator.gen_dataset(train_path, size=train_size, start_index=train_start)
            added_train = train_size

        if self.generate_synth_val:
            val_start = self.next_csv_index(val_path)
            data_generator.gen_dataset(val_path, size=val_size, start_index=val_start)
            added_val = val_size

        print(
            f"Synthetic append: existing train={real_train_count}, existing val={real_val_count}; "
            f"added train={added_train}, added val={added_val}"
        )

    def setup(self, stage=None) -> None:
        dataset = XPSDataset(self.train_data)

        has_val = self.has_val(self.val_data)
        if has_val:
            self.train_dataset = dataset
            self.val_dataset = XPSDataset(self.val_data)
            print(
                f"Using external validation data: train={len(self.train_dataset)} "
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

    def train_dataloader(self) -> DataLoader:
        batch_size = self.train_params.get("batch_size", 128)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        batch_size = self.train_params.get("batch_size", 128)
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def seeded_generator(self, seed) -> torch.Generator:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return torch.Generator().manual_seed(seed)

    def has_val(self, val_data_dir) -> bool:
        if not val_data_dir:
            return False
        val_path = Path(val_data_dir)
        return val_path.exists() and any(val_path.glob("*.csv"))

    def csv_count(self, data_dir) -> int:
        if not data_dir:
            return 0
        p = Path(data_dir)
        if not p.exists():
            return 0
        return sum(1 for _ in p.glob("*.csv"))

    def next_csv_index(self, data_dir) -> int:
        p = Path(data_dir)
        if not p.exists():
            return 0
        indices = [int(f.stem) for f in p.glob("*.csv")]
        if not indices:
            return 0
        return max(indices) + 1
