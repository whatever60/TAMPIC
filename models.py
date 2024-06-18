import torch
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
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
        self.metric_train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.metric_vals_acc = nn.ModuleList(
            [
                torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
                torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
            ]
        )
        self.metric_vals_cm = nn.ModuleList(
            [
                torchmetrics.classification.MulticlassConfusionMatrix(
                    num_classes=num_classes
                ),
                torchmetrics.classification.MulticlassConfusionMatrix(
                    num_classes=num_classes
                ),
            ]
        )

    def forward(self, data, wavelengths):
        return self.model(data, wavelengths)

    def training_step(self, batch, batch_idx):
        data = batch["data"]
        label = batch["label"]
        logits = self(data, self.wavelengths.to(self.device))
        loss = F.cross_entropy(logits, label)
        # log loss and accuracy
        preds = logits.argmax(dim=1)
        self.metric_train_acc(preds, label)
        self.log(
            "train_acc",
            self.metric_train_acc,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data = batch["data"]
        label = batch["label"]
        logits = self(data, self.wavelengths.to(self.device))
        loss = F.cross_entropy(logits, label)
        preds = logits.argmax(dim=1)
        metric_val_acc = self.metric_vals_acc[dataloader_idx]
        metric_val_cm = self.metric_vals_cm[dataloader_idx]
        metric_name = ["val_easy_{}", "val_mid_{}"][dataloader_idx]
        metric_val_acc(preds, label)
        metric_val_cm(preds, label)
        self.log(
            metric_name.format("acc"),
            metric_val_acc,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            metric_name.format("loss"),
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return {
            "loss": loss,
            "preds": preds,
            "targets": label,
            "dataloader_idx": dataloader_idx,
        }

    def on_validation_epoch_end(self) -> None:
        try:
            logger = self.logger
        except AttributeError:
            return
        num_val_sets = len(self.metric_vals_acc)
        fig, ax = plt.subplots(1, num_val_sets, figsize=(10 * num_val_sets, 10))
        for i in range(num_val_sets):
            cm = self.metric_vals_cm[i].compute().detach().cpu().numpy()
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
        # remove the figure from memory
        plt.close(fig)
        # reset the confusion matrix (I shouldn't have to do this, but it seems to be necessary)
        for i in range(num_val_sets):
            self.metric_vals_cm[i].reset()

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
