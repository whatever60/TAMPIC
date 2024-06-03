from torch.optim import AdamW
from torch import functional as F
import pytorch_lightning as pl

from schedulers import WarmupScheduler
from resnet import resnet18_custom, resnet34_custom


class CustomResNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes=1000,
        learning_rate=1e-3,
        warmup_steps=1000,
        total_steps=10000,
        pretrained=True,
        model_type="resnet18",
    ):
        super(CustomResNetLightningModule, self).__init__()
        if model_type == "resnet18":
            self.model = resnet18_custom(num_classes=num_classes, pretrained=pretrained)
        elif model_type == "resnet34":
            self.model = resnet34_custom(num_classes=num_classes, pretrained=pretrained)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pretrained = pretrained

    def forward(self, x_rgb_red, x_rgb_white, x_hsi, wavelengths):
        return self.model(x_rgb_red, x_rgb_white, x_hsi, wavelengths)

    def training_step(self, batch, batch_idx):
        x_rgb_red, x_rgb_white, x_hsi, wavelengths, y = batch
        logits = self(x_rgb_red, x_rgb_white, x_hsi, wavelengths)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Separate parameters into pretrained and newly initialized
        pretrained_params = []
        new_params = []
        pretrained_params, new_params = self.model
        optimizer = AdamW(
            [
                {"params": pretrained_params, "lr": self.learning_rate / 10},
                {"params": new_params, "lr": self.learning_rate},
            ]
        )

        scheduler = WarmupScheduler(
            optimizer,
            T_max=self.total_steps,
            T_warmup=self.warmup_steps,
            start_lr=self.learning_rate / 20,
            peak_lr=self.learning_rate,
            end_lr=self.learning_rate / 5,
            mode="cosine",
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


# Example usage:
# model = CustomResNetLightningModule(num_classes=10, pretrained=True, model_type='resnet18')
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader, val_dataloader)
