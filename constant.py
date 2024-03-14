import os
from pathlib import Path


DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data")
DATA_BASE_DIR = Path(DATA_BASE_DIR)

VQA_DATA_DIR = DATA_BASE_DIR / 'remake_dataset'
VQA_TRAIN = VQA_DATA_DIR / 'dataset/combined_and_shuffled_train.json'
VQA_VAL = VQA_DATA_DIR / 'dataset/combined_and_shuffled_valid.json'
VQA_TEST = VQA_DATA_DIR / 'dataset/combined_and_shuffled_test.json'
VQA_GEN_DIR = DATA_BASE_DIR / '../evaluation/res'



