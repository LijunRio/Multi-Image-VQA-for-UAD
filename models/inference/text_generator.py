import os
import sys
project = 'VQA-UAD'  # root dir
sys.path.append(os.getcwd().split(project)[0] + project)
print(os.getcwd().split(project)[0] + project)


import datetime
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything

from mydatasets.data_module import DataModule
from mydatasets.caption_dataset import CaptionDataset, caption_collate_fn
from mydatasets.transforms import DataTransforms
from models.inference.captioner import Captioner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cli_main():
    parser = ArgumentParser("VQA Inference")
    parser.add_argument("--img_encoder", type=str, default="vit_base")
    parser.add_argument("--ckpt_path", type=str,default="../checkpoints/2023_12_08_12_13_51/epoch=25-step=311.ckpt")
    parser.add_argument("--dataset", type=str, default="vqa")
    parser.add_argument("--image_type", type=str, default="Orig_Black_Black")
    parser.add_argument("--image_channel", type=str, default="1")
    parser.add_argument("--model_type", type=str, default="KQFormer_concat", help="options: 'MI_average_res', 'MI_average_vit', 'MI_concat', 'MI_channel_res', "
                                                                                  "'MI_channel_vit', 'KQFormer_channel'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1) # need to keep the bach size=1
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--data_pct", type=float, default=1)
    parser.add_argument("--prompt", type=str, default='Can you describe the anomaly present sdain the image?')
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--train_dataset", type=str, default='/home/june/Code/VLPv2/data/remake_dataset/dataset/combined_and_shuffled_train.json')
    parser.add_argument("--valid_dataset", type=str, default='/home/june/Code/VLPv2/data/remake_dataset/dataset/combined_and_shuffled_valid.json')
    parser.add_argument("--test_dataset", type=str, default='/home/june/Code/VLPv2/data/remake_dataset/dataset/combined_and_shuffled_test.json')
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # if args.model_type =='SQ_former': from models.VQA.VQA_KQFormer_Concate import VQA
    # elif args.model_type =='average_VIT' or args.model_type == 'average_ResNet': from models.VQA.VQA_MI_average import VQA
    # elif args.model_type == 'concate_VIT_ResNet':from models.VQA.VQA_MI_concate import VQA
    # elif args.model_type =='channel_VIT' or args.model_type == 'channel_ResNet':from models.VQA.VQA_MI_channel import VQA
    # elif args.model_type =='channel_qformer':from models.VQA.VQA_KQFormer_Channel import VQA
    # else: raise ValueError('please enter a valid model type!')
    # print(args.model_type, type(args.model_type))
    if args.model_type =='KQFormer_concat': from models.VQA.VQA_KQFormer_Concate import VQA
    elif args.model_type =='MI_average_vit' or args.model_type =='MI_average_res': from models.VQA.VQA_MI_average import VQA
    elif args.model_type == 'MI_concat':from models.VQA.VQA_MI_concate import VQA
    elif args.model_type =='MI_channel_vit' or args.model_type =='MI_channel_res':from models.VQA.VQA_MI_channel import VQA
    elif args.model_type =='KQFormer_channel':from models.VQA.VQA_KQFormer_Channel import VQA
    else:
        # print(args.model_type)
        raise ValueError('please enter a valid model type!')

    # args.deterministic = True
    args.max_epochs = 1

    seed_everything(args.seed)

    datamodule = DataModule(dataset=CaptionDataset,
                            collate_fn=caption_collate_fn,
                            transforms=DataTransforms,
                            data_pct=args.data_pct,
                            train_dataset = args.train_dataset,
                            valid_dataset = args.valid_dataset,
                            test_dataset = args.test_dataset,
                            batch_size = args.batch_size,
                            num_workers=args.num_workers)

    # model = MGCA.load_from_checkpoint(args.ckpt_path, strict=True)
    model = VQA.load_from_checkpoint(args.ckpt_path, strict=True)  # 加载backbone和预训练参数

    model = Captioner(model, **args.__dict__)
    trainer = Trainer.from_argparse_args(args=args)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":

    cli_main()
