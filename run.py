import os
import subprocess


# current_directory = os.path.dirname(__file__)
# os.chdir(current_directory)

TRAIN_PTH = "/home/june/Code/VQA-UAD/data/dataset/annotations/new_train.json"
VALID_PTH = "/home/june/Code/VQA-UAD/data/dataset/annotations/new_valid.json"
TEST_PTH = "/home/june/Code/VQA-UAD/data/dataset/annotations/new_test.json"
PTH9 = "./data/ckpts/KQF_concat_vit/best.ckpt"

command = [
    "python",
    "./models/inference/text_generator.py",
    "--gpus", "1",
    "--image_type", "Orig_Anomaly_Recon",
    "--img_encoder", "vit_base",
    "--model_type", "KQFormer_concat",
    "--ckpt_path", PTH9,
    "--train_dataset", TRAIN_PTH,
    "--valid_dataset", VALID_PTH,
    "--test_dataset", TEST_PTH
]

subprocess.run(command)
