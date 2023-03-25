import pytorch_lightning as pl
from torchmetrics import JaccardIndex
import torch


class Trainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.jaccard = JaccardIndex(task="multiclass", num_classes=6)

    def training_step(self,batch,batch_index):
        x,y=batch
        y_hat=self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        iou = self.jaccard(y_hat, y[:, 1, :, :])
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        iou = self.jaccard(y_hat, y[:, 1, :, :])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer





