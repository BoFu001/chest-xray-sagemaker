# SageMaker MLOps — End-to-End ML Workflow on AWS

A transfer-learning image classifier (cats vs. dogs) built as a **production-style
machine-learning pipeline on Amazon SageMaker**, rather than a single notebook.
The model is trained in the cloud on a GPU instance, every run is tracked in MLflow,
and the full lifecycle — data → S3 → managed training → experiment tracking →
model artifact → (deployment next) — is driven by reproducible scripts.

**Stack:** AWS SageMaker · MLflow (SageMaker serverless MLflow) · TensorFlow 2.14 (Keras) ·
VGG16 transfer learning · Amazon S3 · IAM · Python

---

## Why this project

Most coursework trains models locally in Colab/Jupyter. This project re-implements
a transfer-learning classifier the way it would run in a company: training is
dispatched to the cloud, data and model artifacts live in S3, experiments are tracked
in MLflow, access is governed by an IAM execution role, and every step is a versioned,
repeatable script. The dataset is deliberately simple and clean so the focus stays on
the AWS / MLOps workflow.

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
                                              └─ params + per-epoch metrics --> MLflow App
                                              └─ model.tar.gz --> S3
```

- **`config.py`** — central config: region, IAM execution-role ARN, MLflow App ARN,
  S3 prefixes.
- **`01_download_data.py`** — pull the Kaggle competition data into `data/`.
- **`02_prepare_data.py`** — sort images into `cats/` and `dogs/` by filename, take
  10,000 per class, and make a **stratified 70/15/15 train/val/test split**
  (reproducible, seed=42). `test` is held out and never touched during training.
- **`03_train.py`** — *launcher* that runs locally and submits a SageMaker training
  job (chooses the instance, framework container, hyperparameters, S3 I/O, and passes
  the MLflow tracking URI and experiment name as environment variables).
- **`src/train.py`** — the *training script* that actually runs inside the cloud
  container. An **end-to-end model**: raw pixels → VGG16 preprocessing → frozen
  VGG16 convolutional base → flatten → feature-wise standardisation → classifier
  head → sigmoid probability. Baking the preprocessing into the model means the
  future endpoint accepts raw images. The standardisation statistics are computed on
  the **training set only** (no leakage from validation/test) and stored inside the
  model, so training and serving can never drift apart. Seeds are fixed (including
  the dataset shuffle) and GPU ops forced deterministic, so runs are reproducible.
- **`src/requirements.txt`** — pinned MLflow client + `sagemaker-mlflow` plugin,
  installed into the training container at startup.

## Experiment tracking

Each training job logs to a SageMaker serverless MLflow App, explicitly:

- **Parameters** (once, before training): `epochs`, `batch_size`, `learning_rate`.
- **Metrics** (every epoch, via a Keras callback): `train_loss`, `train_accuracy`,
  `val_loss`, `val_accuracy`, with step = epoch number starting at 1.
- Each MLflow run is **named after its SageMaker training job**, so a run can be traced
  straight to its job logs, code snapshot, and `model.tar.gz` in S3.

`mlflow.tensorflow.autolog()` was tried first and dropped: it reads the training
dataset before `fit()` to fingerprint it, which silently shifts the shuffle order and
changes the results; it also logs dozens of irrelevant or duplicated parameters and
metrics, and saves model checkpoints by default.

## Dataset

[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) (Kaggle competition) —
25,000 labelled images. Images are **not** committed; run `01_download_data.py`.
After splitting (balanced): train 14,000 · val 3,000 · test 3,000.

## Results

One change at a time, 3 epochs, learning rate 0.001, batch size 32, ml.g4dn.xlarge:

| Model | Val accuracy | Val loss | Training time |
|---|---|---|---|
| VGG16 (frozen) + Flatten + Dense(1) | 97.90% | 0.988 | 435 s |
| + feature-wise standardisation | 97.87% | **0.108** | 551 s |

**Reproducibility is verified**: two independent training jobs of the first configuration
produced identical metrics to every digit.

- **Without standardisation** the raw VGG16 features (25,088 dimensions, unbounded
  ReLU outputs) drive the logits to extremes: accuracy is high, but the few wrong
  predictions are made with near-100% confidence, so validation loss (0.99) is worse
  than random guessing (0.69) and rises every epoch.
- **Standardising the features** cuts validation loss ~9× and makes it decrease over
  epochs, with the same accuracy. Training loss, however, drops to 0.002 (train
  accuracy 99.96%): the head is now memorising the training set — the target for the
  regularization step below.

Changing only the data shuffle order moved validation accuracy by ~0.4%, which is
why every source of randomness is pinned: otherwise run-to-run noise would be
indistinguishable from the effect of a real change.

## How to run

Environment:

```bash
conda create -n sagemaker-mlops python=3.11 -y
conda activate sagemaker-mlops
pip install "sagemaker<3" boto3 kaggle tensorflow scikit-learn
```

Credentials go in `~/.aws/credentials` (AWS) and `~/.kaggle/kaggle.json` (Kaggle).
Accept the competition rules once on Kaggle before downloading.
Create an MLflow App in SageMaker Studio once and put its ARN in `config.py`.

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

- [x] Experiment tracking with SageMaker serverless MLflow
- [x] Feature-wise standardisation of the VGG16 features
- [ ] Regularization pass (dropout, data augmentation, early stopping) vs. baseline
- [ ] Automatic hyperparameter tuning (SageMaker `HyperparameterTuner`)
- [ ] Deploy a real-time inference **endpoint** + test client, evaluated on the held-out test set
- [ ] Model monitoring and data-drift detection

---

*Built as a hands-on study of the SageMaker production ML workflow.*
