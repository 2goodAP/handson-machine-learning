# %%
import gc
from pathlib import Path

import tensorflow as tf
from tensorflow import data, keras

ROOT_DIR = Path().absolute().parent
MLRUNS_DIR = ROOT_DIR.parents[1] / "mlruns"
DATA_DIR = ROOT_DIR / "dataset"
TFR_DIR = DATA_DIR / "tfrecords"
PROTO_DIR = ROOT_DIR / "protobufs"

if not TFR_DIR.is_dir():
    TFR_DIR.mkdir(parents=True)
if not PROTO_DIR.is_dir():
    PROTO_DIR.mkdir(parents=True)

print(f"{MLRUNS_DIR}\n{DATA_DIR}")

# %%
import mlflow

mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR}/mlflow.db")
mlflow.set_experiment("tf_data_api")

# %% [markdown]
# ## 9

# %% [markdown]
# ```proto
# syntax = "proto3";
#
# message BytesList { repeated bytes value = 1; }
# message FloatList { repeated float value = 1 [packed = true]; }
# message Int64List { repeated int64 value = 1 [packed = true]; }
# message Feature {
#     oneof kind {
#         BytesList bytes_list = 1;
#         FloatList float_list = 2;
#         Int64List int64_list = 3;
#     }
# };
# message Features { map<string, Feature> feature = 1; };
# message Example { Features features = 1; };
# ```

# %%
from typing import Generator

import numpy as np
from tensorflow.train import BytesList, Example, Feature, Features, Int64List


def fashion_mnist_to_tfrecord(
    data: np.ndarray,
    labels: np.ndarray,
    record_dir: Path = TFR_DIR / "fashion_mnist",
    n_chunks: int = 10,
    seed: int = 42,
) -> None:
    def __break_dataset_to_chunks(
        data: np.ndarray, labels: np.ndarray, n_chunks: int, seed: int
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        shuf_idx = np.random.default_rng(seed=seed).permutation(len(labels))
        chunk_size = len(labels) // n_chunks

        return (
            (
                data[(idx := shuf_idx[i * chunk_size : (i + 1) * chunk_size])],
                labels[idx],
            )
            for i in range(n_chunks)
        )

    def __fashion_mnist_example(image: np.ndarray | tf.Tensor, label: str) -> Example:
        return Example(
            features=Features(
                feature={
                    "image": Feature(
                        bytes_list=BytesList(
                            value=[
                                tf.io.serialize_tensor(image.astype("float32")).numpy()
                            ]
                        )
                    ),
                    "label": Feature(int64_list=Int64List(value=[int(label)])),
                }
            )
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=seed, stratify=target
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, random_state=seed, stratify=y_train_full
    )
    del X_train_full, y_train_full

    splits = {}
    for split_name, split_data, split_labels in zip(
        ("train", "val", "test"), (X_train, X_val, X_test), (y_train, y_val, y_test)
    ):
        splits[split_name] = __break_dataset_to_chunks(
            split_data, split_labels, n_chunks, seed
        )

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

    record_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        (record_dir / name).mkdir(exist_ok=True)

        for i, (imgs, lbls) in enumerate(split):
            with tf.io.TFRecordWriter(
                str(record_dir / name / f"fashion_mnist_{i:03}.tfrecord")
            ) as rec_file:
                for img, lbl in zip(imgs, lbls):
                    rec_file.write(
                        __fashion_mnist_example(img, lbl).SerializeToString()
                    )


# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

images, targets = (
    (fmnist := fetch_openml(name="Fashion-MNIST", as_frame=False, parser="auto")).data,
    fmnist.target,
)

fashion_mnist_to_tfrecord(images, targets)

# %%
from glob import glob

from tensorflow.io import FixedLenFeature, VarLenFeature


def parse_fashion_mnist_tfrecord(record: bytes) -> tuple[tf.Tensor, tf.Tensor]:
    parsed = tf.io.parse_example(
        record,
        features={
            "image": VarLenFeature(dtype=tf.string),
            "label": FixedLenFeature(shape=(), dtype=tf.int64),
        },
    )

    return (
        tf.io.parse_tensor(parsed["image"].values[0], out_type=tf.float32),
        tf.cast(parsed["label"], dtype=tf.uint8),
    )


train_set = data.TFRecordDataset(
    glob(str(TFR_DIR / "fashion_mnist" / "train" / "fashion_mnist_*.tfrecord")),
    num_parallel_reads=data.AUTOTUNE,
).map(parse_fashion_mnist_tfrecord)

val_set = data.TFRecordDataset(
    glob(str(TFR_DIR / "fashion_mnist" / "val" / "fashion_mnist_*.tfrecord")),
    num_parallel_reads=data.AUTOTUNE,
).map(parse_fashion_mnist_tfrecord)

test_set = data.TFRecordDataset(
    glob(str(TFR_DIR / "fashion_mnist" / "test" / "fashion_mnist_*.tfrecord")),
    num_parallel_reads=data.AUTOTUNE,
).map(parse_fashion_mnist_tfrecord)

# %%
for trs in train_set.batch(32).take(1):
    print("Train:")
    print(trs)

for vs in val_set.batch(32).take(1):
    print("\nVal:")
    print(vs)

for tes in test_set.batch(32).take(1):
    print("\nTest:")
    print(tes)
