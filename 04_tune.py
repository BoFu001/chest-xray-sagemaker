"""Step 4 (run locally): launch a SageMaker hyperparameter tuning job.

Submits the job and returns immediately (wait=False); the tuning runs in the cloud.
Each trial is a normal run of src/train.py. The tuner reads best_val_loss from the training log.

Run: python 04_tune.py
"""
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

from config import REGION, EXECUTION_ROLE_ARN, S3_DATA_PREFIX, S3_OUTPUT_PREFIX, MLFLOW_APP_ARN

sess = sagemaker.Session(boto3.Session(region_name=REGION))
bucket = sess.default_bucket()
data_uri = f"s3://{bucket}/{S3_DATA_PREFIX}"
output_uri = f"s3://{bucket}/{S3_OUTPUT_PREFIX}"

estimator = TensorFlow(
    entry_point="train.py",
    source_dir="src",
    role=EXECUTION_ROLE_ARN,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.14",
    py_version="py310",
    model_dir=False,
    max_run=2400,                    # per-trial time cap in seconds, guards against runaway cost
    hyperparameters={                # fixed values; the tuner adds the three searched ones
        "epochs": 30,
        "batch-size": 32,
        "patience": 3,
    },
    environment={
        "MLFLOW_TRACKING_URI": MLFLOW_APP_ARN,
        "MLFLOW_EXPERIMENT_NAME": "cats-vs-dogs",
    },
    output_path=output_uri,
    sagemaker_session=sess,
)

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name="best_val_loss",
    objective_type="Minimize",
    metric_definitions=[{"Name": "best_val_loss", "Regex": r"best_val_loss=([0-9\.]+)"}],
    hyperparameter_ranges={
        "learning-rate": ContinuousParameter(1e-5, 1e-3, scaling_type="Logarithmic"),
        "hidden-units": IntegerParameter(8, 64),
        "dropout": ContinuousParameter(0.0, 0.5),
    },
    strategy="Bayesian",
    max_jobs=15,
    max_parallel_jobs=1,             # Bayesian learns from each finished trial; also our GPU quota
    early_stopping_type="Off",       # each trial already early-stops on its own
    random_seed=42,
    base_tuning_job_name="cats-vs-dogs-tune",
)

tuner.fit(
    {"train": f"{data_uri}/train", "validation": f"{data_uri}/val"},
    wait=False,
)

print("Tuning job submitted:", tuner.latest_tuning_job.name)
