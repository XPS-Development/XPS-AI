from lightning.pytorch.cli import LightningCLI
from model.train.lightning_module import XPSLightningModule
from model.train.lightning_data import XPSDataModule


def main():
    LightningCLI(
        XPSLightningModule,
        XPSDataModule,
        seed_everything_default=42,
        save_config_callback=None,
        run=True,
    )


if __name__ == "__main__":
    main()
