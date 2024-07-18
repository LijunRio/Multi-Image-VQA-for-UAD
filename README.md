
# Language Models Meet Anomaly Detection for Better Interpretability and Generalizability

This repository hosts the code for our paper titled "[Language Models Meet Anomaly Detection for Better Interpretability and Generalizability](https://arxiv.org/pdf/2404.07622.pdf)", which can also be explored further on our [project page](https://lijunrio.github.io/Multi-Image-VQA-for-UAD/).

## Citation
Please cite our paper if you find this repository helpful for your research:

```latex
@misc{li2024multiimage,
      title={Multi-Image Visual Question Answering for Unsupervised Anomaly Detection}, 
      author={Jun Li and Cosmin I. Bercea and Philip Müller and Lina Felsner and Suhwan Kim and Daniel Rueckert and Benedikt Wiestler and Julia A. Schnabel},
      year={2024},
      eprint={2404.07622},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Dataset Setup
- **MI-VQA Dataset**: Download from [this link](https://drive.google.com/file/d/1mHjtd_yV6ewRFC7ujwBM9_HDSnKUx5l5/view?usp=sharing) and save it to `./data/dataset`.
  - To preprocess the dataset, run:
    ```
    cd data
    python preprocess_dataset.py
    ```

## Usage Instructions
- **Model Training**
  - Train the model by navigating to the model's directory and executing the provided script:
    ```
    cd ./models/VQA
    sh run.sh
    ```
  - Training checkpoints will be saved in `./data/ckpts/`.

- **Result Generation**
  - Generate results by running the inference script:
    ```
    cd ./models/inference
    sh run_vqa_inference.sh
    ```
  - Results will be stored in `./evaluation/res/`.

- **Result Evaluation**
  - Evaluate the results using:
    ```
    cd evaluation
    python evaluate.py
    ```

- **GUI Interface**
  - For a graphical interface, use Streamlit:
    ```
    cd ./models/inference
    streamlit run streamlit_gui.py
    ```


