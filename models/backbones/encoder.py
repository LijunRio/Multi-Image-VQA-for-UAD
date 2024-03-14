import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers import BertConfig, BertTokenizer, logging

from models.backbones import cnn_backbones
from models.backbones.med import BertModel
from models.backbones.vits import create_vit

logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        if "vit" in model_name:
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = model_name[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.model.load_state_dict(state_dict,strict=False)

            self.embed_0 = GlobalEmbedding(
                vision_width, hidden_dim, output_dim
            )

            self.embed_1 = LocalEmbedding(
                vision_width, hidden_dim, vision_width
            )

        else:
            model_function = getattr(
                cnn_backbones, model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )

            # Average pooling
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            self.embed_0 = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

            self.embed_1 = LocalEmbedding(
                self.interm_feature_dim, hidden_dim, 768
            )

    def resnet_forward(self, x, get_local=True):
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # (batch_size, 64, 75, 75)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x

        x = self.pool(x) # (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)

        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        return x, local_features.contiguous()

    def vit_forward(self, x):
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat = self.vit_forward(x)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()  # 返回编码后的CLS、词元特征，contiguous类似于copy


class BertEncoder(nn.Module):
    def __init__(self,
                 tokenizer: BertTokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 freeze_bert: bool = True,
                 # split='text_encoder'
                 ):
        super(BertEncoder, self).__init__()
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True
        self.bert_type = "pritamdeka/BioBert-PubMed200kRCT"
        self.config = BertConfig.from_json_file(
            os.path.join(BASE_DIR, "../../configs/bert2_config.json"))
        self.model = BertModel.from_pretrained(
            self.bert_type,
            config=self.config,
            add_pooling_layer=False,
        )
        self.embed_0 = LocalEmbedding(self.embedding_dim, hidden_dim, self.output_dim)
        self.embed_1 = LocalEmbedding(self.embedding_dim, hidden_dim, self.embedding_dim)

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, ids=None, attn_mask=None, position_ids=None, get_local=False, encoder_hidden_states=None,
                inputs_embeds=None):
        if encoder_hidden_states != None:
            mode = 'multimodal'
        else:
            mode = 'text'
        outputs = self.model(input_ids=ids, inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                             position_ids=position_ids, encoder_hidden_states=encoder_hidden_states,
                             return_dict=True, mode=mode)

        all_feat = outputs.last_hidden_state.unsqueeze(1)

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()

        return report_feat, word_feat

