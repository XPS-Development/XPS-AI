import os
import csv
from datetime import datetime
import lightning.pytorch as pl
import torch
from clearml import Task


class ClearMLCallback(pl.Callback):
    def __init__(
        self,
        project="XPS-Peak-Detection",
        task_name="Lightning-Training",
        monitor="val_loss",
        file_name="best_model.pt",
        artifact_name="best_model",
        csv_name=None,
    ):
        super().__init__()
        self.task = Task.init(project_name=project, task_name=task_name)
        self.logger = self.task.get_logger()

        self.monitor = monitor
        self.file_name = file_name
        self.artifact_name = artifact_name
        self.best = None

        if csv_name:
            self.csv_name = csv_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_name = f"metrics_clearml_{ts}.csv"

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        if (epoch + 1) % 10 != 0:
            return

        self.log_pair(metrics, epoch, "loss", "train_loss", "val_loss")

        self.log_peak_max(metrics, epoch, "iou",
                          "train_iou_peak", "val_iou_peak",
                          "train_iou_max", "val_iou_max")

        self.log_peak_max(metrics, epoch, "precision",
                          "train_prec_peak", "val_prec_peak",
                          "train_prec_max", "val_prec_max")

        self.log_peak_max(metrics, epoch, "recall",
                          "train_rec_peak", "val_rec_peak",
                          "train_rec_max", "val_rec_max")

        self.log_peak_max(metrics, epoch, "accuracy",
                          "train_acc_peak", "val_acc_peak",
                          "train_acc_max", "val_acc_max")

        self.write_csv(trainer, epoch, metrics)

    def on_fit_end(self, trainer, pl_module):
        self.upload_csv(trainer)
        self.save_best(trainer, pl_module, trainer.callback_metrics)

    def log_pair(self, metrics, epoch, title, train_key, val_key):
        train_val = metrics.get(train_key)
        val_val = metrics.get(val_key)
        if train_val is not None:
            self.logger.report_scalar(title, "train", iteration=epoch, value=float(train_val))
        if val_val is not None:
            self.logger.report_scalar(title, "val", iteration=epoch, value=float(val_val))

    def log_peak_max(self, metrics, epoch, title, t_peak, v_peak, t_max, v_max):
        t_peak_val = metrics.get(t_peak)
        v_peak_val = metrics.get(v_peak)
        t_max_val = metrics.get(t_max)
        v_max_val = metrics.get(v_max)

        if t_peak_val is not None:
            self.logger.report_scalar(title, "train_peak", iteration=epoch, value=float(t_peak_val))
        if v_peak_val is not None:
            self.logger.report_scalar(title, "val_peak", iteration=epoch, value=float(v_peak_val))
        if t_max_val is not None:
            self.logger.report_scalar(title, "train_max", iteration=epoch, value=float(t_max_val))
        if v_max_val is not None:
            self.logger.report_scalar(title, "val_max", iteration=epoch, value=float(v_max_val))

    def write_csv(self, trainer, epoch, metrics):
        path = os.path.join(trainer.default_root_dir or ".", self.csv_name)
        new_file = not os.path.exists(path)
        keys = [
            "train_loss", "val_loss",
            "train_iou_peak", "val_iou_peak",
            "train_iou_max", "val_iou_max",
            "train_prec_peak", "val_prec_peak",
            "train_prec_max", "val_prec_max",
            "train_rec_peak", "val_rec_peak",
            "train_rec_max", "val_rec_max",
            "train_acc_peak", "val_acc_peak",
            "train_acc_max", "val_acc_max",
        ]

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["epoch"] + keys)
            values = []
            for k in keys:
                v = metrics.get(k)
                values.append(float(v) if v is not None else "")
            writer.writerow([epoch] + values)

    def upload_csv(self, trainer):
        path = os.path.join(trainer.default_root_dir or ".", self.csv_name)
        if os.path.exists(path):
            self.task.upload_artifact(name="metrics_csv", artifact_object=path)

    def save_best(self, trainer, pl_module, metrics):
        current = metrics.get(self.monitor)
        if current is None:
            return
        current = float(current)
        if self.best is None or current < self.best:
            self.best = current
            path = os.path.join(trainer.default_root_dir or ".", self.file_name)
            torch.save(pl_module.state_dict(), path)
            self.task.upload_artifact(name=self.artifact_name, artifact_object=path)
