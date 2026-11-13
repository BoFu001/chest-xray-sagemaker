"""Step 3 (run locally): launch a SageMaker cloud training job.

This script runs on your machine but trains nothing itself — it only:
  1. packages src/train.py and uploads it to S3
  2. asks SageMaker to spin up a GPU instance (ml.g4dn.xlarge)
  3. that instance pulls data + script from S3, runs train.py, saves the model
     back to S3, and shuts itself down
Locally you only watch the logs.

Run: python 03_train.py
"""
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow

from config import REGION, EXECUTION_ROLE_ARN, S3_DATA_PREFIX, S3_OUTPUT_PREFIX

sess = sagemaker.Session(boto3.Session(region_name=REGION))
bucket = sess.default_bucket()                       # sagemaker-eu-central-1-766577086453
data_uri = f"s3://{bucket}/{S3_DATA_PREFIX}"         # .../chest-xray-pneumonia/data
output_uri = f"s3://{bucket}/{S3_OUTPUT_PREFIX}"     # models land here

estimator = TensorFlow(
    entry_point="train.py",          # script to run in the cloud
    source_dir="src",                # the whole src/ gets packaged and uploaded
    role=EXECUTION_ROLE_ARN,         # role the training instance assumes (created earlier)
    instance_type="ml.g4dn.xlarge",  # one T4 GPU, a cheap GPU box in eu-central-1
    instance_count=1,
    framework_version="2.14",        # SageMaker official TensorFlow container version
    py_version="py310",
    model_dir=False,                 # don't auto-inject --model_dir (we use our own)
    hyperparameters={
        "epochs": 3,
        "batch-size": 32,
    },
    output_path=output_uri,
    base_job_name="chest-xray-vgg16",
    sagemaker_session=sess,
)

# Two data channels: train / validation, mapping to two S3 "folders"
estimator.fit(
    {
        "train": f"{data_uri}/train",
        "validation": f"{data_uri}/val",
    }
)

print("\nTraining complete. Model saved to:", estimator.model_data)
