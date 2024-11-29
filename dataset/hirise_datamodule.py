import albumentations as A
import os
import random
import lightning.pytorch as pl
from torch.utils.data import DataLoader, RandomSampler
from dataset.hirise_dataset import HiRISEDataset


class HiRISEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 4

        self.transform = A.Compose(
            [A.HorizontalFlip(), A.VerticalFlip()]
        )

    def setup(self, stage):
        self.uahirise_train = HiRISEDataset(self.data_dir, stage="train", transform=self.transform)
        self.uahirise_val = HiRISEDataset(self.data_dir, stage="val")
        self.uahirise_test = HiRISEDataset(self.data_dir, stage="test")

    def train_dataloader(self):
        num_samples = len(self.uahirise_train) // 4
        sampler = RandomSampler(self.uahirise_train, num_samples=num_samples)
        return DataLoader(self.uahirise_train, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.uahirise_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.uahirise_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)