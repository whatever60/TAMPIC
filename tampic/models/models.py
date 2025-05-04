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

from ..schedulers import WarmupScheduler
from .resnet import resnet_tampic


class TAMPICResNetLightningModule(L.LightningModule):
    def __init__(
        self,
        model_type: str = "resnet18",
        *,
        hsi_conv_type: str = "identity",
        hsi_avg_dim: int | None = None,
        wavelengths: np.ndarray,
        pretrained: bool = True,
        lr: float,
        warmup_steps: int,
        total_steps: int,
        batch_size: int,
        _pretrained_hsi_base: bool = False,
        _norm_and_sum: bool = True,
        _prediction_log_dir: str | None = None,
        _multi_level_aux_loss_weight: float = 0.0,
        _multi_level_label_clean2idx: dict[str, dict[str, int]] | None = None,
        label_clean2idx: dict[str, int] | None = None,
        # deprecated args
        num_classes: int | None = None,
        _multi_level_num_classes: list[int] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        depth_map = {"resnet18": 18, "resnet34": 34, "resnet50": 50}
        if model_type not in depth_map:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model = resnet_tampic(
            depth=depth_map[model_type],
            num_classes=len(label_clean2idx),
            pretrained=pretrained,
            hsi_conv_type=hsi_conv_type,
            hsi_avg_dim=len(wavelengths) if hsi_avg_dim is None else hsi_avg_dim,
            _pretrained_hsi_base=_pretrained_hsi_base,
            _norm_and_sum=_norm_and_sum,
        )

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.wavelengths = torch.from_numpy(wavelengths).float() / 1000 * 2 - 1
        # self.wavelengths = self.register_buffer(
        #     "wavelengths",
        #     torch.from_numpy(wavelengths).float() / 1000 * 2 - 1,
        # )

        # logging the prediction of the model, but using metrics to implement the logic
        if _prediction_log_dir is None:
            return
        os.makedirs(_prediction_log_dir, exist_ok=True)
        self._prediction_log_dir = _prediction_log_dir

        device = self.device.index
        if device is None:
            device = 0
        self._device_idx = device
        # self.device_str = f"{self.device.type}:{device}"

        self._add_aux_head()
        self._add_metrics()

    def _add_aux_head(self) -> None:
        if self.hparams._multi_level_aux_loss_weight > 0:
            # self._multi_level_num_classes = self.hparams._multi_level_num_classes
            # self._multi_level_aux_loss_weight = (
            #     self.hparams._multi_level_aux_loss_weight
            # )
            self.fc_aux = nn.ModuleList(
                [
                    nn.Linear(self.model.fc.in_features, n_classes)
                    for n_classes in self.hparams._multi_level_num_classes
                ]
            )
        else:
            self.fc_aux = None


    def _compute_aux_loss(
        self, logits_aux: list[torch.Tensor], labels_aux: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary loss.

        Args:
            logits_aux: List of tensors, each (batch_size, n_classes) for each label.
            labels_aux: (batch_size, num_labels_aux)

        Returns:
            Mean auxiliary loss.
        """
        return torch.stack(
            [
                F.cross_entropy(logits_aux[i], labels_aux[:, i])
                for i in range(len(logits_aux))
            ]
        ).mean()

    def _add_metrics(self) -> None:
        num_classes = len(self.hparams.label_clean2idx)
        metrics_acc = MetricCollection(
            {
                "top1_acc": MulticlassAccuracy(num_classes=num_classes, top_k=1),
                "top3_acc": MulticlassAccuracy(num_classes=num_classes, top_k=3),
            }
        )
        self.metrics_acc_train = metrics_acc.clone(prefix="train_")
        self.metrics_acc_val_easy = metrics_acc.clone(prefix="val-easy_")
        self.metrics_acc_val_mid = metrics_acc.clone(prefix="val-mid_")
        self.metric_cm_val_easy = MulticlassConfusionMatrix(num_classes=num_classes)
        self.metric_cm_val_mid = MulticlassConfusionMatrix(num_classes=num_classes)

        logger_prediction = nn.ModuleDict(
            {
                "epoch": CatMetric(),
                "batch_idx": CatMetric(),
                "device": CatMetric(),
                "index_in_df": CatMetric(),
                "label": CatMetric(),
                "embedding": CatMetric(),
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
    def forward(self, data, wavelengths) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(data, wavelengths, _return_embedding=True)

    def training_step(self, batch, batch_idx):
        data = batch["data"]
        label = batch["label"]
        logits, embedding = self(
            data,
            self.wavelengths.unsqueeze(0).expand(label.size(0), -1).to(self.device),
        )
        loss = F.cross_entropy(logits, label)
        self.metrics_acc_train.update(logits, label)
        self.log_dict(
            self.metrics_acc_train,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        if self.fc_aux is not None:
            labels_aux = batch["label_all_levels"]  # [batch_size, num_labels_aux]
            logits_aux = [fc(embedding) for fc in self.fc_aux]
            loss_aux = self._compute_aux_loss(logits_aux, labels_aux)
            self.log(
                "train_loss_all_levels",
                loss_aux,
                on_step=True,
                on_epoch=True,
                # prog_bar=True,
                batch_size=self.batch_size,
            )
        else:
            loss_aux = 0.0

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self._update_prediction_on_step_end(
            dataset="train",
            batch_idx=batch_idx,
            label=label,
            embedding=embedding,
            logits=logits,
            meta=batch["meta"],
        )

        return loss + loss_aux * self.hparams._multi_level_aux_loss_weight

    def _update_prediction_on_step_end(
        self,
        dataset: str,
        batch_idx: int,
        embedding: torch.Tensor,
        label: torch.Tensor,
        logits: torch.Tensor,
        meta: dict,
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
        logger_prediction["embedding"].update(embedding.detach())
        logger_prediction["logits"].update(logits.detach())
        logger_prediction["index_in_df"].update(meta["index_in_df"])
        # logger_prediction["project_id"].update(meta["project_id"])
        # logger_prediction["plate_id"].update(meta["plate_id"])
        # logger_prediction["sample_id"].update(meta["sample_id"])

    def _save_prediction_on_epoch_end(self, dataset: str) -> None:
        if dataset == "train":
            logger_prediction = self.logger_prediction_train
            data_df = self.trainer.train_dataloader.dataset.df.copy()
            prefix = "train"
        elif dataset == "val-easy":
            logger_prediction = self.logger_prediction_val_easy
            data_df = self.trainer.val_dataloaders[0].dataset.df.copy()
            prefix = "val-easy"
        elif dataset == "val-mid":
            logger_prediction = self.logger_prediction_val_mid
            data_df = self.trainer.val_dataloaders[1].dataset.df.copy()
            prefix = "val-mid"
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        if not logger_prediction["embedding"].update_called:
            # no data was logged, so don't save anything
            return
        embedding = logger_prediction["embedding"].compute().cpu().numpy()
        logits = logger_prediction["logits"].compute().cpu().numpy()
        indexs = logger_prediction["index_in_df"].compute().cpu().numpy().astype(int)
        embedding_df = pd.DataFrame(
            embedding,
            index=indexs,
            columns=[f"embedding_{i}" for i in range(embedding.shape[1])],
        )
        try:
            idx2label_clean = self.trainer.datamodule.idx2label_clean
            columns = [idx2label_clean[i] for i in range(len(idx2label_clean))]
        except AttributeError:
            columns = np.arange(logits.shape[1])
        logits_df = pd.DataFrame(logits, columns=columns, index=indexs)
        other_info_df = pd.DataFrame(
            {
                "epoch": logger_prediction["epoch"].compute().cpu().numpy().astype(int),
                "batch_idx": logger_prediction["batch_idx"]
                .compute()
                .cpu()
                .numpy()
                .astype(int),
                "device": logger_prediction["device"]
                .compute()
                .cpu()
                .numpy()
                .astype(int),
                "label": logger_prediction["label"].compute().cpu().numpy().astype(int),
                # "project_id": logger_prediction["project_id"].compute(),
                # "plate_id": logger_prediction["plate_id"].compute(),
                # "sample_id": logger_prediction["sample_id"].compute(),
            },
            index=indexs,
        )
        embedding_df.to_parquet(
            f"{self._prediction_log_dir}/{prefix}_epoch-{self.current_epoch}_embedding.parquet"
        )
        logits_df.to_parquet(
            f"{self._prediction_log_dir}/{prefix}_epoch-{self.current_epoch}_logits.parquet"
        )
        other_info_df.to_csv(
            f"{self._prediction_log_dir}/{prefix}_epoch-{self.current_epoch}_info.csv"
        )
        data_df.to_csv(
            f"{self._prediction_log_dir}/{prefix}_epoch-{self.current_epoch}_df.csv"
        )
        for m in logger_prediction.values():
            m.reset()

    def on_train_start(self) -> None:
        """Save dataset df datasets for all training and validation datasets."""
        df_train = self.trainer.datamodule.df_train_all.copy()
        df_val_easy = self.trainer.datamodule.df_val_easy.copy()
        df_val_mid = self.trainer.datamodule.df_val_mid.copy()
        # put _ at the start of file name so that they are at the top of the folder
        df_train.to_csv(f"{self._prediction_log_dir}/_train_all.csv")
        df_val_easy.to_csv(f"{self._prediction_log_dir}/_val-easy.csv")
        df_val_mid.to_csv(f"{self._prediction_log_dir}/_val-mid.csv")

    def on_train_epoch_end(self) -> None:
        self._save_prediction_on_epoch_end("train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data = batch["data"]
        label = batch["label"]
        logits, embedding = self(data, self.wavelengths)
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

        if self.fc_aux is not None:
            labels_aux = batch["label_all_levels"]
            logits_aux = [fc(embedding) for fc in self.fc_aux]
            loss_aux = self._compute_aux_loss(logits_aux, labels_aux)
            self.log(
                metric_name.format("loss_all_levels"),
                loss_aux,
                on_step=False,
                on_epoch=True,
                # prog_bar=True,
                batch_size=self.batch_size,
            )
        else:
            loss_aux = 0.0

        self.log(
            metric_name.format("loss"),
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        self._update_prediction_on_step_end(
            dataset=["val_easy", "val_mid"][dataloader_idx],
            batch_idx=batch_idx,
            label=label,
            embedding=embedding,
            logits=logits,
            meta=batch["meta"],
        )

        return loss + loss_aux * self.hparams._multi_level_aux_loss_weight

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

    def get_param_groups(self) -> list[dict]:
        pretrained_params, new_params = self.model.get_param_groups()
        params = [
            {"params": pretrained_params, "lr": self.lr / 10},
            {"params": new_params, "lr": self.lr},
        ]
        if self.hparams._multi_level_aux_loss_weight > 0:
            params.append({"params": self.fc_aux.parameters(), "lr": self.lr})
        return params

    def configure_optimizers(self):
        # Separate parameters into pretrained and newly initialized
        params = self.get_param_groups()
        optimizer = AdamW(params)
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
