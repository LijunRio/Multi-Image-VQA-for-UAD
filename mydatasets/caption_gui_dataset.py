import os
import sys
project = 'VQA-UAD'  # root dir
sys.path.append(os.getcwd().split(project)[0] + project)
print(os.getcwd().split(project)[0] + project)
import json
import torch
import torch.utils.data as data
from constant import *
from transformers import GPT2Tokenizer, TFGPT2Model
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CaptionDataset(data.Dataset):
    def __init__(self, image_pth, prompt
                 ,transform=None, imsize=256, caption_max_words=100, split='train', data_pct=1.0, train_dataset='False', valid_dataset='False', test_dataset='False'):
        super().__init__()
        print('test_datset: ', test_dataset, '--------------------vqa_dataset.py')
        # with open(test_dataset, 'r', encoding='utf-8') as f:
        #     self.vqa_data = json.load(f)
        #     print('Size of Test Set：', len(self.vqa_data))

        self.vqa_data = [[image_pth,prompt, '']]
        print(self.vqa_data, '--------------------------')
        self.transform = transform
        self.imsize = imsize
        self.gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.caption_max_words = caption_max_words

    def __len__(self):
        return len(self.vqa_data)

    def encode_instruction(self, sent, prompt):
        caption_tokens = self.gpttokenizer(
            sent,
            return_tensors="pt",  # 返回pytorch tensor
            truncation=True,  # 将过长的句子截断到最大长度
            padding="max_length",  # 将过短的句子填充到最大长度
            max_length=self.caption_max_words,
        )
        prompt_tokens = self.gpttokenizer.encode(prompt)[:-1]

        return prompt_tokens, caption_tokens['attention_mask'][0]

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

        img0 = self.get_img(img0_path)
        img1 = self.get_img(img1_path)
        img2 = self.get_img(img2_path)


        prompt_txt = self.vqa_data[idx][1]
        caption_txt = self.vqa_data[idx][2]
        self.gpttokenizer.pad_token = self.gpttokenizer.eos_token
        caption = self.gpttokenizer(
            caption_txt,
            return_tensors="pt",  # 返回pytorch tensor
            truncation=True,  # 将过长的句子截断到最大长度
            padding="max_length",  # 将过短的句子填充到最大长度
            max_length=self.caption_max_words,
        )


        return img0, img1, img2, filename, caption['input_ids'][0], prompt_txt

def caption_collate_fn(batch):
    """sort sequence"""
    img0s, img1s, img2s, filenames, captions, prompts = [], [], [], [], [], []
    for b in batch:
        img0, img1, img2, filename, caption, prompt = b
        img0s.append(img0)
        img1s.append(img1)
        img2s.append(img2)
        filenames.append(filename)
        captions.append(caption)
        prompts.append(prompt)

    # stack
    img0s = torch.stack(img0s)
    img1s = torch.stack(img1s)
    img2s = torch.stack(img2s)
    captions = torch.stack(captions)



    return_dict = {
        'img0': img0s,
        'img1': img1s,
        'img2': img2s,
        'filename': filenames,
        'caption': captions,
        'prompt':prompts
    }
    return return_dict



