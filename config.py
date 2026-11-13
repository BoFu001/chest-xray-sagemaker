"""Central configuration — every script reads AWS / S3 / data settings from here."""

# ---------- AWS ----------
REGION = "eu-central-1"

# Execution role that SageMaker training jobs / endpoints assume at runtime
# to read/write S3. (Created in the IAM console: AWS service -> SageMaker -> Execution)
EXECUTION_ROLE_ARN = "arn:aws:iam::766577086453:role/AmazonSageMaker-ExecutionRole-chestxray"

# ---------- S3 layout ----------
# SageMaker uses a default bucket: sagemaker-eu-central-1-766577086453
S3_PREFIX        = "chest-xray-pneumonia"        # top-level "folder" in the bucket
S3_DATA_PREFIX   = f"{S3_PREFIX}/data"           # images live here
S3_OUTPUT_PREFIX = f"{S3_PREFIX}/output"         # trained models land here

# ---------- Local / data ----------
LOCAL_DATA_DIR = "data"                          # downloaded & prepared images
CLASS_NAMES    = ["NORMAL", "PNEUMONIA"]         # used by 02_prepare_data.py when splitting
