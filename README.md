# Chest X-Ray Pneumonia Classification — End-to-End on AWS SageMaker

Binary image classification (NORMAL vs. PNEUMONIA) built as a **production-style
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

## Pipeline

```
 local machine                         AWS cloud
 ------------                          ---------
 01_download_data.py  --(Kaggle API)-->  local data/
 02_prepare_data.py   --(re-split)----->  data/prepared/
                       --(aws s3 sync)-->  S3  (chest-xray-pneumonia/data)
 03_train.py (launcher) ----------------->  SageMaker Training Job
                                              └─ ml.g4dn.xlarge (T4 GPU)
                                              └─ runs src/train.py
                                              └─ model.tar.gz --> S3
```

- **`config.py`** — central config: region, IAM execution-role ARN, S3 prefixes.
- **`01_download_data.py`** — pull the dataset from Kaggle into `data/`.
- **`02_prepare_data.py`** — the original `val` split has only 16 images; merge it
  back into `train` and carve out a **stratified 15% validation set** (reproducible,
  seed=42). `test` (624 images) is held out and never touched during training.
- **`03_train.py`** — *launcher* that runs locally and submits a SageMaker training
  job (chooses the instance, framework container, hyperparameters, and S3 I/O).
- **`src/train.py`** — the *training script* that actually runs inside the cloud
  container. An **end-to-end model**: raw pixels → VGG16 preprocessing → frozen
  VGG16 convolutional base → classifier head → sigmoid probability. Baking the
  preprocessing into the model means the future endpoint accepts raw images.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— 5,856 images, 2 classes. Images are **not** committed; run `01_download_data.py`.
After re-splitting: train 4,446 · val 786 · test 624.

> Note: the training set is imbalanced (~3:1 PNEUMONIA:NORMAL), so accuracy alone
> is misleading — precision/recall on the held-out test set are the next metrics to add.

## Baseline results

| Model | Val accuracy |
|---|---|
| VGG16 (frozen) + Flatten + Dense(1), 3 epochs | **98.35%** |

A deliberately minimal baseline. See the roadmap for planned improvements.

## How to run

```bash
# 1. Environment
conda create -n chest-xray python=3.11 -y
conda activate chest-xray
pip install "sagemaker<3" boto3 kaggle tensorflow scikit-learn

# 2. Credentials: ~/.aws/credentials (AWS) and ~/.kaggle/kaggle.json (Kaggle)
# 3. Pipeline
python 01_download_data.py      # download dataset
python 02_prepare_data.py       # re-split into data/prepared/
aws s3 sync data/prepared s3://<your-bucket>/chest-xray-pneumonia/data
python 03_train.py              # launch cloud training
```

A GPU training quota (`ml.g4dn.xlarge for training job usage`) must be requested
once via AWS Service Quotas for a new account.

## Roadmap

- [ ] Evaluate on the held-out test set: precision / recall / confusion matrix
- [ ] Regularization pass (dropout, data augmentation, early stopping) vs. baseline
- [ ] Automatic hyperparameter tuning (SageMaker `HyperparameterTuner`)
- [ ] Deploy a real-time inference **endpoint** + test client
- [ ] Model monitoring and data-drift detection

---

*Built as a hands-on study of the SageMaker production ML workflow.*
