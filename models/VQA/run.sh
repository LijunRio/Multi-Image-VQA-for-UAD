
#TRAIN_PTH=/media/june/B6CAFC57CAFC14F9/VLPv2/dataset/annotations/train.json
#VALID_PTH=/media/june/B6CAFC57CAFC14F9/VLPv2/dataset/annotations/valid.json
#TEST_PTH=/media/june/B6CAFC57CAFC14F9/VLPv2/dataset/annotations/test.json

TRAIN_PTH=/home/june/Code/VQA-UAD/data/dataset/annotations/new_train.json
VALID_PTH=/home/june/Code/VQA-UAD/data/dataset/annotations/new_valid.json
TEST_PTH=/home/june/Code/VQA-UAD/data/dataset/annotations/new_test.json


CUDA_VISIBLE_DEVICES=0 python VQA_KQFormer_Concate.py --gpus 1 --strategy ddp  --img_encoder vit_base --image_type Orig_Anomaly_Recon --batch_size 48 --max_epochs 40 --learning_rate 1.5e-5 --patience 10 --num_workers 6 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
#CUDA_VISIBLE_DEVICES=0 python VQA_KQFormer_Channel.py --gpus 1 --strategy ddp  --img_encoder vit_base --image_channel 3 --batch_size 48 --max_epochs 40 --learning_rate 1.5e-5 --patience 10 --num_workers 6 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
#CUDA_VISIBLE_DEVICES=0 python VQA_MI_average.py --gpus 1 --strategy ddp  --img_encoder vit_base --image_type Orig_Anomaly_Recon --batch_size 48 --max_epochs 40 --learning_rate 1.5e-5 --patience 10 --num_workers 6 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
#CUDA_VISIBLE_DEVICES=0 python VQA_MI_concate.py --gpus 1 --strategy ddp  --img_encoder vit_base --image_type Orig_Anomaly_Recon --batch_size 48 --max_epochs 40 --learning_rate 1.5e-5 --patience 10 --num_workers 6 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
#CUDA_VISIBLE_DEVICES=0 python VQA_MI_channel.py --gpus 1 --strategy ddp  --img_encoder vit_base --image_channel 3 --batch_size 48 --max_epochs 40 --learning_rate 1.5e-5 --patience 10 --num_workers 6 --train_dataset $TRAIN_PTH --valid_dataset $VALID_PTH --test_dataset $TEST_PTH
