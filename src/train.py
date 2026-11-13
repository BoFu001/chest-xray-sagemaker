"""VGG16 transfer learning, pneumonia binary classification.
Baseline + held-out test-set evaluation (precision / recall / confusion matrix).

SageMaker conventions: data is read from SM_CHANNEL_*, the model is saved to SM_MODEL_DIR.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # NORMAL=0, PNEUMONIA=1 (positive class)


def load(data_dir, batch_size, shuffle=True):
    """Read {data_dir}/{NORMAL,PNEUMONIA}/*.jpeg and resize to 224x224."""
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def main():
    # Fully reproducible experiments
    tf.keras.utils.set_random_seed(42)          # Python / NumPy / TensorFlow seeds
    tf.config.experimental.enable_op_determinism()  # force deterministic GPU ops

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST"))
    args = p.parse_args()

    train_ds = load(args.train, args.batch_size, shuffle=True)
    val_ds = load(args.validation, args.batch_size, shuffle=False)

    # End-to-end model: raw pixels -> VGG16 preprocessing -> VGG16 (frozen) -> flatten -> probability
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(preprocess_input)(inputs)
    base = VGG16(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False                                  # freeze; train only the output layer below
    x = layers.Flatten()(base.output)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)

    # ---- Evaluate on the held-out test set (never seen during training) ----
    if args.test:
        test_ds = load(args.test, args.batch_size, shuffle=False)   # keep order aligned
        y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int).flatten()
        y_prob = model.predict(test_ds, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n==== Test set evaluation ====")
        print("Confusion matrix (rows=true, cols=pred), classes =", CLASS_NAMES)
        print(confusion_matrix(y_true, y_pred))
        print()
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Save as a TF SavedModel (under version dir "1", for later deployment)
    model.save(os.path.join(args.model_dir, "1"))
    print("Model saved.")


if __name__ == "__main__":
    main()
