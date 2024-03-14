# Multi-Image Visual Question Answering for Unsupervised Anomaly Detection

This repository contains the code for the paper "Multi-Image Visual Question Answering for Unsupervised Anomaly Detection".

## Citation
If you find this repo useful for your research, please consider citing our paper:

```latex


```

## Dataset
- Donlad the dataset to ./data/dataset :
  - Data preprocess
    ```
    cd data
    python preprocess_dataset.py
    ```
## Usage
- To train the model

- The checkpoint file will generate in ./data/ckpts/
    ```
    cd ./models/VQA
    sh run.sh
    ``` 
- To generate the results

- The results will generate in ./evaluation/res/
  ```
  cd ./models/inference
  sh run_vqa_inference.sh
  ``` 
- To evaluate the results
    ```
    cd evaluation
    python evaluate.py
    ```
  
# Acknowledge
