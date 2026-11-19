"""VGG16 transfer learning, cats-vs-dogs binary classification — minimal reproducible baseline.

SageMaker conventions: data is read from SM_CHANNEL_*, the model is saved to SM_MODEL_DIR.
"""
import argparse
import os

import mlflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG_SIZE = 224
CLASS_NAMES = ["cats", "dogs"]   # cats=0, dogs=1


def load(data_dir, batch_size, shuffle):
    """Read {data_dir}/{cats,dogs}/*.jpg and resize to 224x224."""
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,                     # fixed shuffle order, independent of global RNG state
    )


def main():
    # Fully reproducible experiments
    tf.keras.utils.set_random_seed(42)              # Python / NumPy / TensorFlow seeds
    tf.config.experimental.enable_op_determinism()  # force deterministic GPU ops

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    args = p.parse_args()

    train_ds = load(args.train, args.batch_size, shuffle=True)
    val_ds = load(args.validation, args.batch_size, shuffle=False)

    # End-to-end model: raw pixels -> VGG16 preprocessing -> VGG16 (frozen) -> flatten -> standardise -> probability
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(preprocess_input)(inputs)
    base = VGG16(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False                                  # freeze; train only the output layer below
    x = layers.Flatten()(base.output)

    # Feature-wise standardisation (like StandardScaler), statistics from the training set only
    norm = layers.Normalization()
    features = Model(inputs, x).predict(load(args.train, args.batch_size, shuffle=False), verbose=0)
    norm.adapt(features)
    x = norm(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Log each epoch's metrics to MLflow under consistent train_/val_ names (step = epoch number, from 1)
    log_epoch = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: mlflow.log_metrics({
            "train_loss": logs["loss"],
            "train_accuracy": logs["accuracy"],
            "val_loss": logs["val_loss"],
            "val_accuracy": logs["val_accuracy"],
        }, step=epoch + 1)
    )

    with mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME")):   # name the run after the SageMaker job
        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate})
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2, callbacks=[log_epoch])

    # Save as a TF SavedModel (under version dir "1", for later deployment)
    model.save(os.path.join(args.model_dir, "1"))
    print("Model saved.")


if __name__ == "__main__":
    main()
