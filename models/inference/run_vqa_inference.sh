#TRAIN_PTH=/media/june/B6CAFC57CAFC14F9/VLPv2/dataset/annotations/train.json
#VALID_PTH=/media/june/B6CAFC57CAFC14F9/VLPv2/dataset/annotations/valid.json
#TEST_PTH=/media/june/B6CAFC57CAFC14F9/VLPv2/dataset/annotations/test.json

TRAIN_PTH=/home/june/Code/VQA-UAD/data/dataset/annotations/new_train.json
VALID_PTH=/home/june/Code/VQA-UAD/data/dataset/annotations/new_valid.json
TEST_PTH=/home/june/Code/VQA-UAD/data/dataset/annotations/new_test.json


PTH1=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/MI_average_res/best.ckpt
PTH2=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/MI_concat_res/best.ckpt
PTH3=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/MI_channel_res/best.ckpt
PTH4=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/KQF_concat_res/best.ckpt
PTH5=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/KQF_channel_res/best.ckpt
PTH6=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/MI_average_vit/best.ckpt
PTH7=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/MI_concat_vit/best.ckpt
PTH8=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/MI_channel_vit/best.ckpt
PTH9=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/KQF_concat_vit/best.ckpt
PTH10=/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/KQF_channel_vit/best.ckpt


CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder resnet_50 --model_type MI_average_res --ckpt_path $PTH1 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder resnet_50 --model_type MI_concat --ckpt_path $PTH2 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder resnet_50 --model_type MI_channel_res --ckpt_path $PTH3 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder resnet_50 --model_type KQFormer_concat --ckpt_path $PTH4 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder resnet_50 --model_type KQFormer_channel --ckpt_path $PTH5 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH


CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder vit_base --model_type MI_average_vit --ckpt_path $PTH6 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder vit_base --model_type MI_concat --ckpt_path $PTH7 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder vit_base --model_type MI_channel_vit --ckpt_path $PTH8 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder vit_base --model_type KQFormer_concat --ckpt_path $PTH9 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
CUDA_VISIBLE_DEVICES=0 python text_generator.py --gpus 1 --image_type Orig_Anomaly_Recon --img_encoder vit_base --model_type KQFormer_channel --ckpt_path $PTH10 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH