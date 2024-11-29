import lightning.pytorch as pl
import torch

from glpdepth.model import GLPDepth
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from utils.criterion import SiLogLoss
from utils.save import save_metrics
from utils.metrics import d1, d2, d3


class Model(pl.LightningModule):
    def __init__(self, max_depth):
        super().__init__()

        self.model = GLPDepth(max_depth=max_depth)

        self.lr = 1e-4
        self.loss_fn = SiLogLoss()

        self.metrics = MetricCollection({
            'd1': d1(),
            'd2': d2(),
            'd3': d3(),
            'mae': MeanAbsoluteError(),
            'rmse': MeanSquaredError(squared=False)
        })

    def forward(self, image):
        pred = self.model(image)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss_step', loss, on_step=True, prog_bar=True)
        self.log('train_loss_epoch', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.metrics.update(y_hat, y)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        result = self.metrics.compute()
        self.log_dict(result)
        save_metrics(self.logger.log_dir, result, self.current_epoch, self.trainer.max_epochs)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.metrics.update(y_hat, y)

    def on_test_epoch_end(self):
        result = self.metrics.compute()
        print(result)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]