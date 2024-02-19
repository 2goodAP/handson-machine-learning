# %% [markdown]
# # Exercises

# %%
import gc
from pathlib import Path

import tensorflow as tf
from tensorflow import data, keras

ROOT_DIR = Path().absolute().parent
MLRUNS_DIR = ROOT_DIR.parents[1] / "mlruns"
DATA_DIR = ROOT_DIR / "dataset"
PROTO_DIR = ROOT_DIR / "protobufs"
TFR_DIR = DATA_DIR / "tfrecords"
IMDB_DIR = DATA_DIR / "large_movie_review"

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
from contextlib import ExitStack

import numpy as np
from tensorflow.train import BytesList, Example, Feature, Features, Int64List


def fashion_mnist_to_tfrecord(
    dataset: data.Dataset,
    name: str,
    record_dir: Path = TFR_DIR / "fashion_mnist",
    n_shards: int = 10,
    seed: int = 42,
) -> list[str]:
    def __fashion_mnist_example(image: np.ndarray | tf.Tensor, label: str) -> Example:
        return Example(
            features=Features(
                feature={
                    "image": Feature(
                        bytes_list=BytesList(
                            value=[tf.io.serialize_tensor(image).numpy()]
                        )
                    ),
                    "label": Feature(int64_list=Int64List(value=[int(label)])),
                }
            )
        )

    (record_dir / name).mkdir(parents=True, exist_ok=True)
    paths = [
        str(record_dir / name / f"{shard:03}.tfrecord") for shard in range(n_shards)
    ]

    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path)) for path in paths]

        for i, (img, lbl) in dataset.enumerate():
            writers[i % n_shards].write(
                __fashion_mnist_example(img, lbl).SerializeToString()
            )

    return paths


# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

BUFFER_SIZE = 10_000
SEED = 42

images, targets = (
    (fmnist := fetch_openml(name="Fashion-MNIST", as_frame=False, parser="auto")).data,
    fmnist.target,
)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    images.reshape(-1, 28, 28).astype("uint8"),
    targets,
    test_size=0.2,
    random_state=SEED,
    stratify=targets,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, random_state=SEED, stratify=y_train_full
)
del X_train_full, y_train_full

train_set, val_set, test_set = (
    data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
        buffer_size=BUFFER_SIZE
    ),
    data.Dataset.from_tensor_slices((X_val, y_val)),
    data.Dataset.from_tensor_slices((X_test, y_test)),
)

del X_train, y_train, X_val, y_val, X_test, y_test
gc.collect()

# %%
train_paths = fashion_mnist_to_tfrecord(train_set, name="train")
val_paths = fashion_mnist_to_tfrecord(val_set, name="validation")
test_paths = fashion_mnist_to_tfrecord(test_set, name="test")

del train_set, val_set, test_set
gc.collect()

# %%
from glob import glob

from tensorflow.io import FixedLenFeature
from tensorflow.keras import layers

BUFFER_SIZE = 10_000
N_THREADS = data.AUTOTUNE


def create_tfrecord_dataset(
    record_paths: list[str],
    batch_size: int = 128,
    n_threads: int | None = N_THREADS,
    cache: bool = False,
    shuffle_buf_size: int | None = None,
    seed: int = 42,
) -> data.TFRecordDataset:
    def __parse_fashion_mnist_tfrecord(record: bytes) -> tuple[tf.Tensor, tf.Tensor]:
        parsed = tf.io.parse_example(
            record,
            features={
                "image": FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
                "label": FixedLenFeature(shape=(), dtype=tf.int64),
            },
        )

        return (
            tf.ensure_shape(
                tf.io.parse_tensor(parsed["image"], out_type=tf.uint8), shape=(28, 28)
            ),
            tf.cast(parsed["label"], dtype=tf.uint8),
        )

    dataset = data.TFRecordDataset(record_paths, num_parallel_reads=n_threads).map(
        __parse_fashion_mnist_tfrecord, num_parallel_calls=n_threads
    )
    if cache:
        dataset = dataset.cache()
    if shuffle_buf_size is not None:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buf_size, seed=SEED, reshuffle_each_iteration=True
        )

    return dataset.batch(
        batch_size, drop_remainder=True, num_parallel_calls=n_threads
    ).prefetch(n_threads)


train_set = create_tfrecord_dataset(
    glob(str(TFR_DIR / "fashion_mnist" / "train" / "*.tfrecord")),
    shuffle_buf_size=BUFFER_SIZE,
)

(norm := layers.Normalization(input_shape=train_set.element_spec[0].shape[1:])).adapt(
    train_set.map(lambda X, y: X, num_parallel_calls=N_THREADS)
)

train_set = train_set.map(lambda X, y: (norm(X), y), num_parallel_calls=N_THREADS)

val_set = create_tfrecord_dataset(
    glob(str(TFR_DIR / "fashion_mnist" / "validation" / "*.tfrecord")),
    cache=True,
).map(lambda X, y: (norm(X), y), num_parallel_calls=N_THREADS)

test_set = create_tfrecord_dataset(
    glob(str(TFR_DIR / "fashion_mnist" / "test" / "*.tfrecord")),
    cache=True,
)

# %%
for trs in train_set.take(1):
    print("Train:")
    print(trs)

for vs in val_set.take(1):
    print("\nVal:")
    print(vs)

for tes in test_set.take(1):
    print("\nTest:")
    print(tes)

# %%
model = keras.Sequential(
    [
        layers.Flatten(input_shape=train_set.element_spec[0].shape[1:]),
        layers.Dense(100, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(50, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(50, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="nadam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)
model.summary()

# %%
from tensorflow.keras.callbacks import EarlyStopping

mlflow.tensorflow.autolog()

history = model.fit(
    train_set,
    epochs=1000,
    validation_data=val_set,
    callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
)

# %%
final_model = keras.Sequential([norm, model])
final_model.compile(
    optimizer=model.optimizer, loss=model.loss, metrics=model.metrics[1:]
)

# %%
test_loss, test_accuracy = final_model.evaluate(test_set)

# %% [markdown]
# ## 10

# %%
# import tarfile
# from io import BytesIO

# import requests

# if not IMDB_DIR.is_dir():
#     IMDB_DIR.mkdir(parents=True, exist_ok=True)

# with BytesIO(
#     initial_bytes=requests.get(
#         "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
#         allow_redirects=True,
#     ).content
# ) as archive:
#     tar = tarfile.open(fileobj=archive, mode="r:gz")
#     tar.extractall(IMDB_DIR)

# %% [markdown]
# ### For Val & Test Sets
#
# - 7,500 from both `pos` and `neg` for **Val**
# - 5,000 from both `pos` and `neg` for **Test**

# %%
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow import data

SHUF_BUF_SIZE = 25_000
CYCLE_LENGTH = 1000
N_THREADS = 8
SEED = 42
VAL_SIZE = 7500

TRAIN_DIR = IMDB_DIR / "aclImdb" / "train"
TEST_DIR = IMDB_DIR / "aclImdb" / "test"


def create_imdb_dataset(
    pos_paths: list[str],
    neg_paths: list[str],
    cycle_len: int = CYCLE_LENGTH,
    n_threads: int = N_THREADS,
    shuf_buf_size: int | None = None,
    batch_size: int = 64,
) -> data.Dataset:
    dataset = (
        data.Dataset.list_files(pos_paths)
        .interleave(
            lambda filename: data.TextLineDataset(
                filename, num_parallel_reads=n_threads
            ).map(
                lambda line: (line, tf.constant(1, dtype=tf.uint8)),
                num_parallel_calls=n_threads,
            ),
            cycle_length=CYCLE_LENGTH,
            num_parallel_calls=n_threads,
        )
        .concatenate(
            data.Dataset.list_files(neg_paths).interleave(
                lambda filename: data.TextLineDataset(
                    filename, num_parallel_reads=n_threads
                ).map(
                    lambda line: (line, tf.constant(0, dtype=tf.uint8)),
                    num_parallel_calls=n_threads,
                ),
                cycle_length=cycle_len,
                num_parallel_calls=n_threads,
            )
        )
    )

    return (
        (
            dataset.shuffle(buffer_size=shuf_buf_size, reshuffle_each_iteration=True)
            if shuf_buf_size is not None
            else dataset
        )
        .batch(batch_size, num_parallel_calls=n_threads)
        .prefetch(data.AUTOTUNE)
    )


val_test_pos_paths, val_test_neg_paths = (
    np.fromiter(glob(str(TEST_DIR / "pos" / "*.txt")), dtype="object"),
    np.fromiter(glob(str(TEST_DIR / "neg" / "*.txt")), dtype="object"),
)

shuf_idx = np.random.default_rng().permutation(len(val_test_pos_paths))
val_pos_paths, val_neg_paths = (
    val_test_pos_paths[shuf_idx[:VAL_SIZE]],
    val_test_neg_paths[shuf_idx[:VAL_SIZE]],
)
test_pos_paths, test_neg_paths = (
    val_test_pos_paths[shuf_idx[VAL_SIZE:]],
    val_test_neg_paths[shuf_idx[VAL_SIZE:]],
)

del val_test_pos_paths, val_test_neg_paths

train_set = create_imdb_dataset(
    pos_paths=glob(str(TRAIN_DIR / "pos" / "*.txt")),
    neg_paths=glob(str(TRAIN_DIR / "neg" / "*.txt")),
    shuf_buf_size=SHUF_BUF_SIZE,
)
val_set = create_imdb_dataset(pos_paths=val_pos_paths, neg_paths=val_neg_paths)
test_set = create_imdb_dataset(pos_paths=test_pos_paths, neg_paths=test_neg_paths)

del val_pos_paths, val_neg_paths, test_pos_paths, test_neg_paths
gc.collect()

# %%
from tensorflow.keras import layers

text_vec = layers.TextVectorization()
text_vec.adapt(train_set.take(5000).map(lambda X, y: X, num_parallel_calls=N_THREADS))

train_set = train_set.map(lambda X, y: (text_vec(X), y), num_parallel_calls=N_THREADS)
val_set = val_set.map(lambda X, y: (text_vec(X), y), num_parallel_calls=N_THREADS)

# %%
DROPOUT_RATE = 0.5
EMBED_OUT = 50


def compute_sentence_embeddings(word_embeds: tf.Tensor, sum_axis: int = 1) -> tf.Tensor:
    n_words = tf.math.count_nonzero(
        tf.math.count_nonzero(word_embeds, axis=-1), axis=-1, keepdims=True
    )

    return tf.reduce_sum(word_embeds, axis=sum_axis) / tf.sqrt(
        tf.cast(n_words, dtype=word_embeds.dtype)
    )


sentiment_model = keras.models.Sequential(
    [
        layers.Embedding(input_dim=text_vec.vocabulary_size(), output_dim=EMBED_OUT),
        layers.Lambda(compute_sentence_embeddings),
        layers.Dense(100, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(rate=DROPOUT_RATE),
        layers.Dense(50, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(rate=DROPOUT_RATE),
        layers.Dense(1, activation="sigmoid"),
    ]
)
sentiment_model.compile(
    optimizer="nadam",
    loss="binary_crossentropy",
    metrics=["accuracy", "Precision", "Recall"],
    jit_compile=False,
)

# %%
another_example = tf.constant(
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        [[6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)

compute_sentence_embeddings(another_example)

# %%
from tensorflow.keras.callbacks import EarlyStopping

mlflow.tensorflow.autolog()

history = sentiment_model.fit(
    train_set,
    epochs=20,
    validation_data=val_set,
    callbacks=EarlyStopping(patience=10, restore_best_weights=True),
)

# %%
final_sentiment_model = keras.models.Sequential(
    [layers.Input(shape=(), dtype=tf.string), text_vec, sentiment_model]
)
final_sentiment_model.compile(
    optimizer=sentiment_model.optimizer,
    loss=sentiment_model.loss,
    metrics=sentiment_model.metrics[1:],
)

# %%
test_metrics = final_sentiment_model.evaluate(test_set)

# %% [markdown]
# ### Using TFDS

# %%
import tensorflow_datasets as tfds

train, val, test = tfds.load(
    name="imdb_reviews",
    split=["train", "test[:75%]", "test[75%:]"],
    as_supervised=True,
    shuffle_files=True,
)

# %%
for X, y in train.take(1):
    print(X, y)

print()

for X, y in val.take(1):
    print(X, y)

print()

for X, y in test.take(1):
    print(X, y)
