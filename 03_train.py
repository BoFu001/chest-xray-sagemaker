"""Step 3 (run locally): launch a SageMaker cloud training job.

Minimal launcher — it packages src/train.py, runs it on a GPU instance, and saves the
model back to S3. Local machine only submits the job and watches the logs.

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
    role=EXECUTION_ROLE_ARN,         # role the training instance assumes
    instance_type="ml.g4dn.xlarge",  # one T4 GPU
    instance_count=1,
    framework_version="2.14",        # SageMaker official TensorFlow container
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

# Two data channels: train / validation
estimator.fit(
    {
        "train": f"{data_uri}/train",
        "validation": f"{data_uri}/val",
    }
)

print("\nTraining complete. Model saved to:", estimator.model_data)
