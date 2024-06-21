import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy
from torchmetrics.aggregation import CatMetric
import matplotlib.pyplot as plt
import seaborn as sns

from schedulers import WarmupScheduler
from resnet import resnet18_tampic, resnet34_tampic, resnet50_tampic


class TAMPICResNetLightningModule(L.LightningModule):
    def __init__(
        self,
        model_type="resnet18",
        *,
        pretrained=True,
        num_classes,
        lr,
        warmup_steps,
        total_steps,
        batch_size: int,
        wavelengths,
        _pretrained_hsi_base: bool = False,
        _norm_and_sum: bool = True,
        _prediction_log_dir: str = None,
    ):
        super(TAMPICResNetLightningModule, self).__init__()
        self.save_hyperparameters()
        if model_type == "resnet18":
            self.model = resnet18_tampic(
                num_classes=num_classes,
                pretrained=pretrained,
                num_wavelengths=len(wavelengths),
                _pretrained_hsi_base=_pretrained_hsi_base,
                _norm_and_sum=_norm_and_sum,
            )
        elif model_type == "resnet34":
            self.model = resnet34_tampic(
                num_classes=num_classes,
                pretrained=pretrained,
                num_wavelengths=len(wavelengths),
                _pretrained_hsi_base=_pretrained_hsi_base,
                _norm_and_sum=_norm_and_sum,
            )
        elif model_type == "resnet50":
            self.model = resnet50_tampic(
                num_classes=num_classes,
                pretrained=pretrained,
                num_wavelengths=len(wavelengths),
                _pretrained_hsi_base=_pretrained_hsi_base,
                _norm_and_sum=_norm_and_sum,
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.wavelengths = torch.from_numpy(wavelengths).float() / 1000 * 2 - 1
        metrics_acc = MetricCollection(
            {
                "top1_acc": MulticlassAccuracy(num_classes=num_classes, top_k=1),
                "top3_acc": MulticlassAccuracy(num_classes=num_classes, top_k=3),
            }
        )
        self.metrics_acc_train = metrics_acc.clone(prefix="train_")
        self.metrics_acc_val_easy = metrics_acc.clone(prefix="val-easy")
        self.metrics_acc_val_mid = metrics_acc.clone(prefix="val-mid")
        self.metric_cm_val_easy = MulticlassConfusionMatrix(num_classes=num_classes)
        self.metric_cm_val_mid = MulticlassConfusionMatrix(num_classes=num_classes)

        # logging the prediction of the model, but using metrics to implement the logic
        if _prediction_log_dir is None:
            return
        os.makedirs(_prediction_log_dir, exist_ok=True)
        self._prediction_log_dir = _prediction_log_dir
        logger_prediction = nn.ModuleDict(
            {
                "epoch": CatMetric(),
                "batch_idx": CatMetric(),
                "device": CatMetric(),
                "index_in_df": CatMetric(),
                "label": CatMetric(),
                "logits": CatMetric(),
                "project_id": CatMetric(),
                "plate_id": CatMetric(),
                "sample_id": CatMetric(),
            }
        )
        self.logger_prediction_train = {
            k: v.clone() for k, v in logger_prediction.items()
        }
        self.logger_prediction_val_easy = {
            k: v.clone() for k, v in logger_prediction.items()
        }
        self.logger_prediction_val_mid = {
            k: v.clone() for k, v in logger_prediction.items()
        }

        device = self.device.index
        if device is None:
            device = 0
        self._device_idx = device
        # self.device_str = f"{self.device.type}:{device}"


    def forward(self, data, wavelengths):
        return self.model(data, wavelengths)

    def training_step(self, batch, batch_idx):
        data = batch["data"]
        label = batch["label"]
        print("----------")
        logits = self(data, self.wavelengths.to(self.device))
        print("++++++++++")
        loss = F.cross_entropy(logits, label)
        self.metrics_acc_train.update(logits, label)
        self.log_dict(
            self.metrics_acc_train,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )

        self._update_prediction_on_step_end(
            dataset="train",
            batch_idx=batch_idx,
            label=label,
            logits=logits,
            meta=batch["meta"],
        )
        return loss

    def _update_prediction_on_step_end(
        self, dataset: str, batch_idx, label, logits, meta: dict
    ) -> None:
        if dataset == "train":
            logger_prediction = self.logger_prediction_train
        elif dataset == "val_easy":
            logger_prediction = self.logger_prediction_val_easy
        elif dataset == "val_mid":
            logger_prediction = self.logger_prediction_val_mid
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        batch_size = label.size(0)
        logger_prediction["epoch"].update(torch.full((batch_size,), self.current_epoch))
        logger_prediction["batch_idx"].update(torch.full((batch_size,), batch_idx))
        logger_prediction["device"].update(torch.full((batch_size,), self._device_idx))
        logger_prediction["label"].update(label)
        logger_prediction["logits"].update(logits.detach())
        logger_prediction["index_in_df"].update(meta["index_in_df"])
        # logger_prediction["project_id"].update(meta["project_id"])
        # logger_prediction["plate_id"].update(meta["plate_id"])
        # logger_prediction["sample_id"].update(meta["sample_id"])

    def _save_prediction_on_epoch_end(self, dataset: str) -> None:
        if dataset == "train":
            logger_prediction = self.logger_prediction_train
            prefix = "train"
        elif dataset == "val-easy":
            logger_prediction = self.logger_prediction_val_easy
            prefix = "val-easy"
        elif dataset == "val-mid":
            logger_prediction = self.logger_prediction_val_mid
            prefix = "val-mid"
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        logits = logger_prediction["logits"].compute().cpu().numpy()
        try:
            idx2label_clean = self.trainer.datamodule.idx2label_clean
            columns = [idx2label_clean[i] for i in range(len(idx2label_clean))]
        except AttributeError:
            columns = np.arange(logits.shape[1])
        indexs = logger_prediction["index_in_df"].compute().cpu().numpy()
        logits_df = pd.DataFrame(logits, columns=columns, index=indexs)
        other_info_df = pd.DataFrame(
            {
                "epoch": logger_prediction["epoch"].compute(),
                "label": logger_prediction["label"].compute(),
                # "project_id": logger_prediction["project_id"].compute(),
                # "plate_id": logger_prediction["plate_id"].compute(),
                # "sample_id": logger_prediction["sample_id"].compute(),
            },
            index=indexs,
        )
        logits_df.to_csv(
            f"{self._prediction_log_dir}/{prefix}_logits_epoch_{self.current_epoch}.csv"
        )
        other_info_df.to_csv(
            f"{self._prediction_log_dir}/{prefix}_info_epoch_{self.current_epoch}.csv"
        )
        for m in logger_prediction.values():
            m.reset()

    def on_training_epoch_end(self) -> None:
        self._save_prediction_on_epoch_end("train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data = batch["data"]
        label = batch["label"]
        logits = self(data, self.wavelengths.to(self.device))
        loss = F.cross_entropy(logits, label)
        if dataloader_idx == 0:
            metric_val_acc = self.metrics_acc_val_easy
            metric_val_cm = self.metric_cm_val_easy
        else:
            metric_val_acc = self.metrics_acc_val_mid
            metric_val_cm = self.metric_cm_val_mid
        metric_val_acc.update(logits, label)
        metric_val_cm.update(logits, label)
        metric_name = ["val_easy_{}", "val_mid_{}"][dataloader_idx]
        self.log(
            metric_name.format("loss"),
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        self._update_prediction_on_step_end(
            dataset=["val_easy", "val_mid"][dataloader_idx],
            batch_idx=batch_idx,
            label=label,
            logits=logits,
            meta=batch["meta"],
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        # log scalar metrics. compuate, log, reset
        acc_val_easy = self.metrics_acc_val_easy.compute()
        acc_val_mid = self.metrics_acc_val_mid.compute()
        self.log_dict(acc_val_easy)
        self.log_dict(acc_val_mid)
        self.metrics_acc_val_easy.reset()
        self.metrics_acc_val_mid.reset()

        # log the confusion matrix to tensorboard
        try:
            logger = self.logger
        except AttributeError:
            return

        num_val_sets = 2
        fig, ax = plt.subplots(1, num_val_sets, figsize=(15 * num_val_sets, 15))
        for i, metric_cm in enumerate(
            [self.metric_cm_val_easy, self.metric_cm_val_mid]
        ):
            cm = metric_cm.compute().detach().cpu().numpy()
            # reset the confusion matrix (I shouldn't have to do this, but it seems to be necessary)
            metric_cm.reset()
            # normalize the confusion matrix for each true class (row)
            cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
            sns.heatmap(cm, ax=ax[i], square=True, lw=0.5, annot=True, fmt=".2f")
            ax[i].set_title(["val_easy", "val_mid"][i])
            # set x and y axis tick labels to actual class names and set x label rotation to 45 degrees
            idx2label = self.trainer.datamodule.idx2label_clean
            labels = [idx2label[i] for i in range(len(idx2label))]
            ax[i].set_xticklabels(labels, rotation=45, ha="right")
            ax[i].set_yticklabels(labels, rotation=0)
            ax[i].set_xlabel("Predicted")
            ax[i].set_ylabel("True")
        logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
        plt.close(fig)

        self._save_prediction_on_epoch_end("val-easy")
        self._save_prediction_on_epoch_end("val-mid")

    def configure_optimizers(self):
        # Separate parameters into pretrained and newly initialized
        pretrained_params = []
        new_params = []
        pretrained_params, new_params = self.model.get_param_groups()
        optimizer = AdamW(
            [
                {"params": pretrained_params, "lr": self.lr / 10},
                {"params": new_params, "lr": self.lr},
            ]
        )
        scheduler = WarmupScheduler(
            optimizer,
            T_max=self.total_steps,
            T_warmup=self.warmup_steps,
            start_lr=self.lr / 20,
            peak_lr=self.lr,
            end_lr=self.lr / 5,
            mode="cosine",
        )
        scheduler = {
            "scheduler": scheduler,
            "name": "warmup_scheduler",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


# Example usage:
# model = TAMPICResNetLightningModule(num_classes=10, pretrained=True, model_type='resnet18')
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader, val_dataloader)
