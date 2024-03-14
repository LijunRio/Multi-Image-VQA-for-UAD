import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, train_dataset, valid_dataset, test_dataset,batch_size, num_workers,
                 crop_size=224):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def train_dataloader(self):
        # 图像变换
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            split="train", transform=transform, data_pct=self.data_pct,
            train_dataset=self.train_dataset, valid_dataset=self.valid_dataset, test_dataset=self.test_dataset)  # 数据集初始化

        return DataLoader(
            dataset,
            pin_memory=True,  # 内存足够时，将该项设置为true时可以加快张量转移到gpu的速度
            drop_last=True,  # 抛弃最后一个不完整的batch
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct,
            train_dataset=self.train_dataset, valid_dataset=self.valid_dataset, test_dataset=self.test_dataset)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="test", transform=transform, data_pct=self.data_pct,
            train_dataset=self.train_dataset, valid_dataset=self.valid_dataset, test_dataset=self.test_dataset)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
