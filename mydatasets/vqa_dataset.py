import json
import os

import torch
import torch.utils.data as data
from constant import *
from transformers import BertTokenizer
from transformers import GPT2Tokenizer, TFGPT2Model
import cv2
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, train_dataset='False', valid_dataset='False',test_dataset='False',
                 imsize=256, text_max_words=100):
        super().__init__()
        print('train_dataset: ', train_dataset, '--------------------vqa_dataset.py')
        print('valid_dataset: ', valid_dataset, '--------------------vqa_dataset.py')
        print('test_datset: ', test_dataset, '--------------------vqa_dataset.py')
        if split == 'train':
            with open(train_dataset, 'r', encoding='utf-8') as f:
                self.vqa_data = json.load(f)
            print('Size of training set：', len(self.vqa_data))
        else:
            with open(valid_dataset, 'r', encoding='utf-8') as f:
                self.vqa_data = json.load(f)
                print('Size of test set：', len(self.vqa_data))
        self.transform = transform
        self.imsize = imsize
        self.gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.text_max_words = text_max_words

    def __len__(self):
        return len(self.vqa_data)

    def my_get_instruction(self, sent):
        sent_list = sent.split('<sep>')
        ques = sent_list[0]
        ans = sent_list[1]
        # Set the padding token for the tokenizer
        self.gpttokenizer.pad_token = self.gpttokenizer.eos_token
        tokens_ques = self.gpttokenizer(
            ques,
            return_tensors="pt",  # 返回pytorch tensor
            truncation=True,  # 将过长的句子截断到最大长度
            padding="max_length",  # 将过短的句子填充到最大长度
            max_length=self.text_max_words,
        )
        tokens = self.gpttokenizer(
            ques + ' ' + ans,
            return_tensors="pt",  # 返回pytorch tensor
            truncation=True,  # 将过长的句子截断到最大长度
            padding="max_length",  # 将过短的句子填充到最大长度
            max_length=self.text_max_words,
        )
        # 获取问题长度
        prompt_length = torch.sum(tokens_ques['attention_mask'][0])
        # 获取label，即需要频闭的词，频闭的地方需要为[-1]
        label = tokens['input_ids'][0].clone()
        label[:prompt_length] = -100  # 把token id中问题的部分设置伪-100
        sent_len = torch.sum(tokens['attention_mask'][0])
        # 句子结束即<end>token分后也标为-100
        label[sent_len + 1:] = -100  # 把句子后面的所有token也变成-100. 注意上面的sep_token_id就是句子最后面的<sep>分割复的位置，token 对应为102
        return tokens['input_ids'][0], tokens['attention_mask'][0], label

    def get_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.imsize, self.imsize))
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        filename = self.vqa_data[idx][0]
        img0_path = filename.replace('{}', 'orig')
        img1_path = filename.replace('{}', 'anomaly')
        img2_path = filename.replace('{}', 'rec')
        # print('img0_path: ', img0_path)
        # print('img1_path: ', img1_path)
        # print('img2_path: ', img2_path)
        img0 = self.get_img(img0_path)
        img1 = self.get_img(img1_path)
        img2 = self.get_img(img2_path)

        prompt_txt = self.vqa_data[idx][1]
        caption_txt = self.vqa_data[idx][2]
        # print('prompt: ', prompt_txt)
        # print('caption: ', caption_txt)
        text, attn, label = self.my_get_instruction(prompt_txt + '<sep>' + caption_txt)

        return img0, img1, img2, text, attn, label


def multimodal_collate_fn(batch):
    """sort sequence"""
    img0s, img1s, img2s, texts, attns, labels = [], [], [], [], [], []
    for b in batch:
        img0, img1, img2, text, attn, label = b
        img0s.append(img0)
        img1s.append(img1)
        img2s.append(img2)

        texts.append(text)
        attns.append(attn)
        labels.append(label)

    # stack
    img0s = torch.stack(img0s)
    img1s = torch.stack(img1s)
    img2s = torch.stack(img2s)

    texts = torch.stack(texts)
    attns = torch.stack(attns)
    labels = torch.stack(labels)

    return_dict = {
        'img0': img0s,
        'img1': img1s,
        'img2': img2s,
        'text': texts,
        'attn': attns,
        'label': labels,
    }
    return return_dict
