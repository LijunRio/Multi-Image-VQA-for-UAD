import os
import sys
project = 'VQA-UAD'  # root dir
sys.path.append(os.getcwd().split(project)[0] + project)
print(os.getcwd().split(project)[0] + project)
import torch
from argparse import Namespace
import streamlit as st
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning import seed_everything
# from mydatasets.data_module import DataModule
# from mydatasets.caption_dataset import CaptionDataset, caption_collate_fn
from mydatasets.caption_gui_dataset import CaptionDataset, caption_collate_fn
from mydatasets.data_gui_module import DataModule
from mydatasets.transforms import DataTransforms
from models.inference.captioner import Captioner
from models.VQA.VQA_KQFormer_Concate import VQA
from pytorch_lightning import Trainer, seed_everything


# Define a function to run inference
def run_inference(args):
    # Setting up datamodule
    # Capdataset = CaptionDataset(image_pth=args.image_path, prompt=args.prompt)
    datamodule = DataModule(image_path=args.image_path, prompt=args.prompt,dataset=CaptionDataset,
                            collate_fn=caption_collate_fn,
                            transforms=DataTransforms,
                            data_pct=args.data_pct,
                            train_dataset=args.train_dataset,
                            valid_dataset=args.valid_dataset,
                            test_dataset=args.test_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    # Loading model
    print('successfully load model......')
    print(args)
    # model = Captioner.load_from_checkpoint(args.ckpt_path, strict=True)
    model = VQA.load_from_checkpoint(args.ckpt_path, strict=True)  # 加载backbone和预训练参数
    model = Captioner(model, **vars(args))
    trainer = Trainer.from_argparse_args(args=args)

    # Running inference
    # res = trainer.test(model, datamodule=datamodule)
    res = trainer.predict(model, datamodule=datamodule)
    return res


def main():
    st.title("Multi Image VQA for Unsupervised Anomaly Detection")
    # Sidebar for user input
    st.sidebar.header("Inputs")
    image_path = st.sidebar.text_input("image path","/media/june/B6CAFC57CAFC14F9/Data/Anomaly_dataset/posttreatment_change_25_{}.png")
    # prompt = st.sidebar.text_input("Prompt", "Can you comment on the severity of the pathology?")
    prompt_options = [
        "Is the case normal?",
        "Please describe the condition of the brain.",
        "Can you comment on the severity of the pathology?",
        "Are there areas in the anomaly maps that highlight a normal variation of the healthy, rather than pathological areas (false positives)?",
        "Can you describe the differences highlighted between anomaly maps and origin image and why it is the healthy region?",
        "Is the pseudo-healthy reconstruction a plausible restoration of the input to a healthy state?",
        "Do the anomaly maps accurately reflect the selected disease?"
    ]

    prompt_options.append("Other")

    selected_prompt = st.sidebar.selectbox("Select a prompt or choose 'Other' to enter your own prompt", prompt_options,
                                           index=0)

    if selected_prompt == "Other":
        custom_prompt = st.sidebar.text_input("Enter your own prompt")
        prompt = custom_prompt if custom_prompt else "No custom prompt entered"
    else:
        prompt = selected_prompt

    # fix input
    img_encoder = "vit_base"
    ckpt_path = "/media/june/B6CAFC57CAFC14F9/VLPv2/checkpoints/KQF_concat_vit/best.ckpt"
    dataset = "vqa"
    image_type = "Orig_Anomaly_Recon"
    image_channel = "3"
    model_type = "KQFormer_concat"
    seed =42
    batch_size =  1
    num_workers = 0
    learning_rate = 2e-4
    weight_decay = 0.05
    data_pct = 1.0
    beam_size = 5
    train_dataset = "/home/june/Code/VQA-UAD/data/dataset/annotations/new_train.json"
    valid_dataset = "/home/june/Code/VQA-UAD/data/dataset/annotations/new_valid.json"
    test_dataset ="/home/june/Code/VQA-UAD/data/dataset/annotations/new_test.json"

    # Display the images
    image_path_orig = image_path.replace("{}", "orig")
    image_path_anomaly = image_path.replace("{}", "anomaly")
    image_path_rec = image_path.replace("{}", "rec")

    if os.path.exists(image_path_orig) and os.path.exists(image_path_anomaly) and os.path.exists(image_path_rec):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_path_orig, caption='Original Image', use_column_width=True)

        with col2:
            st.image(image_path_anomaly, caption='Anomaly Image', use_column_width=True)

        with col3:
            st.image(image_path_rec, caption='Reconstructed Image', use_column_width=True)
    else:
        st.write("Images not found at provided paths")

    st.title("Predicted Results:")
    if st.sidebar.button("Run Inference"):
        with st.spinner("Running inference..."):
            args = Namespace(img_encoder=img_encoder, ckpt_path=ckpt_path, dataset=dataset, image_type=image_type,
                             image_channel=image_channel, image_path=image_path, model_type=model_type, seed=seed,
                             batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate,
                             weight_decay=weight_decay, data_pct=data_pct, prompt=prompt, beam_size=beam_size,
                             train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset)

            # Running inference
            res = run_inference(args)

        # Display result
        # st.write(res[0]['res'][0], text_size="100px")
        if not res[0]['res'][0]:
            st.write("Sorry, I don't know. Out of my knowledge.")
        else:
            st.write(res[0]['res'][0], text_size="100px")

if __name__ == "__main__":
    main()
