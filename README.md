# Cats vs. Dogs Classification — End-to-End on AWS SageMaker

Binary image classification (cats vs. dogs) built as a **production-style
machine-learning pipeline on Amazon SageMaker**, rather than a single notebook.
The model is trained in the cloud on a GPU instance and the full lifecycle —
data → S3 → managed training → model artifact → (deployment next) — is driven by
reproducible scripts.

**Stack:** AWS SageMaker · TensorFlow 2.14 (Keras) · VGG16 transfer learning ·
Amazon S3 · IAM · Python

---

## Why this project

Most coursework trains models locally in Colab/Jupyter. This project re-implements
a transfer-learning classifier the way it would run in a company: training is
dispatched to the cloud, data and model artifacts live in S3, access is governed
by an IAM execution role, and every step is a versioned, repeatable script.
The dataset is deliberately simple and clean so the focus stays on the AWS workflow.

## Pipeline

```
 local machine                         AWS cloud
 ------------                          ---------
 01_download_data.py  --(Kaggle API)-->  local data/
 02_prepare_data.py   --(split)-------->  data/prepared/
                       --(aws s3 sync)-->  S3  (cats-vs-dogs/data)
 03_train.py (launcher) ----------------->  SageMaker Training Job
                                              └─ ml.g4dn.xlarge (T4 GPU)
                                              └─ runs src/train.py
                                              └─ model.tar.gz --> S3
```

- **`config.py`** — central config: region, IAM execution-role ARN, S3 prefixes.
- **`01_download_data.py`** — pull the Kaggle competition data into `data/`.
- **`02_prepare_data.py`** — sort images into `cats/` and `dogs/` by filename, take
  10,000 per class, and make a **stratified 70/15/15 train/val/test split**
  (reproducible, seed=42). `test` is held out and never touched during training.
- **`03_train.py`** — *launcher* that runs locally and submits a SageMaker training
  job (chooses the instance, framework container, hyperparameters, and S3 I/O).
- **`src/train.py`** — the *training script* that actually runs inside the cloud
  container. An **end-to-end model**: raw pixels → VGG16 preprocessing → frozen
  VGG16 convolutional base → classifier head → sigmoid probability. Baking the
  preprocessing into the model means the future endpoint accepts raw images.
  Seeds are fixed and GPU ops forced deterministic, so runs are reproducible.

## Dataset

[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) (Kaggle competition) —
25,000 labelled images. Images are **not** committed; run `01_download_data.py`.
After splitting (balanced): train 14,000 · val 3,000 · test 3,000.

## Baseline results

| Model | Epochs | Val accuracy | Training time |
|---|---|---|---|
| VGG16 (frozen) + Flatten + Dense(1) | 3 | **98.33%** | 405 s on ml.g4dn.xlarge |

A deliberately minimal baseline. Validation accuracy is stable, but validation loss
rises every epoch (0.645 → 0.928): the head becomes over-confident on the few images
it gets wrong — the first target for the regularization step below.

## How to run

Environment:

```bash
conda create -n cats-dogs python=3.11 -y
conda activate cats-dogs
pip install "sagemaker<3" boto3 kaggle tensorflow scikit-learn
```

Credentials go in `~/.aws/credentials` (AWS) and `~/.kaggle/kaggle.json` (Kaggle).
Accept the competition rules once on Kaggle before downloading.

Pipeline:

```bash
python 01_download_data.py
python 02_prepare_data.py
aws s3 sync data/prepared s3://<your-bucket>/cats-vs-dogs/data
python 03_train.py
```

A GPU training quota (`ml.g4dn.xlarge for training job usage`) must be requested
once via AWS Service Quotas for a new account.

## Roadmap

- [ ] Regularization pass (dropout, data augmentation, early stopping) vs. baseline
- [ ] Automatic hyperparameter tuning (SageMaker `HyperparameterTuner`)
- [ ] Deploy a real-time inference **endpoint** + test client, evaluated on the held-out test set
- [ ] Model monitoring and data-drift detection

---

*Built as a hands-on study of the SageMaker production ML workflow.*
