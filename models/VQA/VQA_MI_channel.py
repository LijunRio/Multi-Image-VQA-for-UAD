import os
import sys

project = 'VQA-UAD'  # root dir
sys.path.append(os.getcwd().split(project)[0] + project)

from pytorch_lightning import LightningModule, Trainer, seed_everything
from models.backbones.encoder import BertEncoder, ImageEncoder
from transformers import GPT2LMHeadModel, GPT2Config, TextGenerationPipeline
import torch
import torch.nn as nn
import datetime
from dateutil import tz
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from argparse import ArgumentParser
from mydatasets.data_module import DataModule
from mydatasets.transforms import DataTransforms
from mydatasets.vqa_dataset import MultimodalPretrainingDataset, multimodal_collate_fn
import torchvision.transforms as transforms

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class VQA(LightningModule):
    def __init__(self,
                 img_encoder: str = "resnet_50",
                 image_channel: str = 1,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        # init encoders
        self.encoder_name = img_encoder
        self.image_channel = image_channel
        print('Vision_backbone: {}, Image_Channel: {}'.format(img_encoder, image_channel), '--------------------------------')
        self.img_encoder_q = ImageEncoder(model_name=img_encoder, output_dim=self.hparams.emb_dim)
        if img_encoder == 'resnet_50':
            self.embed_img = nn.Sequential(nn.Conv1d(in_channels=362 * 3, out_channels=128, kernel_size=1),
                                           nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1))
            self.embed_img2 = nn.Conv1d(in_channels=1024, out_channels=768, kernel_size=1)
        else:
            self.embed_img = nn.Sequential(nn.Conv1d(in_channels=197 * 3, out_channels=128, kernel_size=1),
                                           nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1))
        self.grayscale_transform = transforms.Grayscale(num_output_channels=1)
        config = GPT2Config.from_json_file(os.path.join(BASE_DIR, "../../configs/gpt2_config.json"))
        self.decoder = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "../backbones/GPT2"), config=config)

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''
        if self.image_channel == '1':
            orig_image = self.grayscale_transform(batch['img0'])
            orig_image = F.pad(orig_image, (0, 0, 0, 0, 0, 2), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(orig_image)
        elif self.image_channel =='2':
            orig_image = self.grayscale_transform(batch['img0'])
            anomaly_image = self.grayscale_transform(batch['img1'])
            combine_images = torch.cat([orig_image, anomaly_image], dim=1)
            combine_images = F.pad(combine_images, (0, 0, 0, 0, 0, 1), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(combine_images)
        elif self.image_channel == '3':
            orig_image = self.grayscale_transform(batch['img0'])
            anomaly_image = self.grayscale_transform(batch['img1'])
            persudo_image = self.grayscale_transform(batch['img2'])
            combine_images = torch.cat([orig_image, anomaly_image, persudo_image], dim=1)
            global_img_feat, local_img_feat = self.img_encoder_q(combine_images)
        elif self.image_channel == 'Anomaly_Black_Black':
            anomaly_image = self.grayscale_transform(batch['img1'])
            anomaly_image = F.pad(anomaly_image, (0, 0, 0, 0, 0, 2), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(anomaly_image)
        elif self.image_channel == 'Orig_Black_Recon':
            orig_image = self.grayscale_transform(batch['img0'])
            persudo_image = self.grayscale_transform(batch['img2'])
            combine_images = torch.cat([orig_image, persudo_image], dim=1)
            combine_images = F.pad(combine_images, (0, 0, 0, 0, 0, 1), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(combine_images)
        else:
            raise ValueError('image channel wrong!')

        img_feat = torch.cat([global_img_feat.unsqueeze(1), local_img_feat], dim=1)  # (b, 197, 768) /(b, 362, 1024)

        if self.encoder_name == 'resnet_50':
            img_feat = img_feat.permute(0, 2, 1)
            img_feat = self.embed_img2(img_feat)
            img_feat = img_feat.permute(0, 2, 1)
        img_feat = F.normalize(img_feat, dim=-1)

        output = self.decoder(input_ids=batch['text'], attention_mask=batch['attn'],
                              encoder_hidden_states=img_feat.contiguous(), labels=batch['label'])
        text_loss = output['loss']
        return text_loss

    def encode(self, img0, img1, img2):
        if self.image_channel == '1':
            orig_image = self.grayscale_transform(img0)
            orig_image = F.pad(orig_image, (0, 0, 0, 0, 0, 2), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(orig_image)
        elif self.image_channel =='2':
            orig_image = self.grayscale_transform(img0)
            anomaly_image = self.grayscale_transform(img1)
            combine_images = torch.cat([orig_image, anomaly_image], dim=1)
            combine_images = F.pad(combine_images, (0, 0, 0, 0, 0, 1), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(combine_images)
        elif self.image_channel == '3':
            orig_image = self.grayscale_transform(img0)
            anomaly_image = self.grayscale_transform(img1)
            persudo_image = self.grayscale_transform(img2)
            combine_images = torch.cat([orig_image, anomaly_image, persudo_image], dim=1)
            global_img_feat, local_img_feat = self.img_encoder_q(combine_images)
        elif self.image_channel == 'Anomaly_Black_Black':
            anomaly_image = self.grayscale_transform(img1)
            anomaly_image = F.pad(anomaly_image, (0, 0, 0, 0, 0, 2), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(anomaly_image)
        elif self.image_channel == 'Orig_Black_Recon':
            orig_image = self.grayscale_transform(img0)
            persudo_image = self.grayscale_transform(img2)
            combine_images = torch.cat([orig_image, persudo_image], dim=1)
            combine_images = F.pad(combine_images, (0, 0, 0, 0, 0, 1), mode='constant', value=0.0)
            global_img_feat, local_img_feat = self.img_encoder_q(combine_images)


        img_feat = torch.cat([global_img_feat.unsqueeze(1), local_img_feat], dim=1)  # (b, 197, 768) /(b, 362, 1024)
        if self.encoder_name == 'resnet_50':
            img_feat = img_feat.permute(0, 2, 1)
            img_feat = self.embed_img2(img_feat)
            img_feat = img_feat.permute(0, 2, 1)
        img_feat = F.normalize(img_feat, dim=-1)
        return img_feat

    def decode(self, input_ids, encoder_output):
        output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)
        return output

    def training_step(self, batch, batch_idx):
        text_loss = self(batch, batch_idx, "train")
        loss = text_loss
        log = {"train_loss": loss}
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text_loss = self(batch, batch_idx, "valid")
        loss = text_loss
        log = {"val_loss": loss, }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        pass
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):  # self.ha
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--image_channel", type=str, default="3")
        parser.add_argument("--emb_dim", type=int,default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=1.5e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=20)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--train_dataset", type=str, default='/home/june/Code/VLPv2/data/remake_dataset/dataset/combined_and_shuffled_train.json')
        parser.add_argument("--valid_dataset", type=str, default='/home/june/Code/VLPv2/data/remake_dataset/dataset/combined_and_shuffled_valid.json')
        parser.add_argument("--test_dataset", type=str, default='/home/june/Code/VLPv2/data/remake_dataset/dataset/combined_and_shuffled_test.json')
        parser.add_argument("--hidden_dim", type=int, default=768)
        parser.add_argument("--query_num", type=int, default=32)
        parser.add_argument("--patience", type=int, default=5)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = VQA.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True

    # seed
    seed_everything(args.seed)
    datamodule = DataModule(dataset=MultimodalPretrainingDataset,
                            collate_fn=multimodal_collate_fn,
                            transforms=DataTransforms,
                            data_pct=args.data_pct,
                            train_dataset = args.train_dataset,
                            valid_dataset = args.valid_dataset,
                            test_dataset = args.test_dataset,
                            batch_size = args.batch_size,
                            num_workers=args.num_workers)

    # Add load from checkpoint
    model = VQA(**args.__dict__)
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"../../data/ckpts/VQA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=args.patience, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
    )

    model.training_steps = model.num_training_steps(trainer, datamodule)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
