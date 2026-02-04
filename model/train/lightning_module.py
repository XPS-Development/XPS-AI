import torch
from torch.optim import Adam
import lightning.pytorch as pl
from model.models.model_deeper import XPSModel
from model.train.metrics import IoU, Accuracy, Precision, Recall, DiceFocalLoss, IoULoss


class XPSLightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate=5e-4,
        dicefocal_weight=5.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = XPSModel()
        self.iouloss = IoULoss()
        self.dicefocal = DiceFocalLoss()

        self.iou = IoU()
        self.acc = Accuracy()
        self.prec = Precision()
        self.rec = Recall()

    def forward(self, x):
        return self.model(x)

    def criterion(self, inp, tar):
        return self.iouloss(inp[:, 0, :], tar[:, 0, :]) + self.hparams.dicefocal_weight * self.dicefocal(
            inp[:, 1, :], tar[:, 1, :]
        )

    def compute_metrics(self, output, target):
        return {
            "iou_peak": self.iou(output[:, 0, :], target[:, 0, :]),
            "acc_peak": self.acc(output[:, 0, :], target[:, 0, :]),
            "prec_peak": self.prec(output[:, 0, :], target[:, 0, :]),
            "rec_peak": self.rec(output[:, 0, :], target[:, 0, :]),
            "iou_max": self.iou(output[:, 1, :], target[:, 1, :]),
            "acc_max": self.acc(output[:, 1, :], target[:, 1, :]),
            "prec_max": self.prec(output[:, 1, :], target[:, 1, :]),
            "rec_max": self.rec(output[:, 1, :], target[:, 1, :]),
        }

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)

        metrics = self.compute_metrics(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)

        metrics = self.compute_metrics(output, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
