# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import gc
from pathlib import Path

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
# ## The `tensorflow.data` API

# %%
import tensorflow as tf
from tensorflow import data

# %%
ts = data.Dataset.from_tensor_slices(tf.range(10))

for t in ts:
    print(t)

# %%
ts[0]

# %%
rs = data.Dataset.range(10)

for r in rs:
    print(r)

# %%
rs[0]

# %%
X_nested = {"a": ([1, 2, 3], [4, 5, 6]), "b": [7, 8, 9]}

for x in Dataset.from_tensor_slices(X_nested):
    print(x)

# %%
X_nested = {"a": [[1, 2, 3], [4, 5, 6]], "b": [[7, 8, 9], [10, 11, 12]]}

for x in Dataset.from_tensor_slices(X_nested):
    print(x)

# %% [markdown]
# ## Chaining Transformations

# %%
import tensorflow as tf
from tensorflow import data

# %%
dataset = data.Dataset.range(10)

for rb in dataset.repeat(5).batch(8):
    print(rb)

# %%
for srb in dataset.shuffle(5, seed=42).repeat(5).batch(10):
    print(srb)

# %%
for d in dataset:  # No change
    print(d)

# %%
for md in dataset.map(lambda x: 2 * x + 4, num_parallel_calls=data.AUTOTUNE):
    print(md)

# %%
for fd in (
    dataset.map(lambda x: (4 * x + 5) // 2, num_parallel_calls=data.AUTOTUNE)
    .repeat(5)
    .batch(8)
    .filter(lambda x: tf.reduce_sum(x) > 80)
):
    print(fd)

# %%
for id in dataset.repeat().batch(7).take(5):  # 5 out of inf
    print(id)

# %% [markdown]
# ## Shuffling the data

# %%
from tensorflow import data

dataset = data.Dataset.range(10)

for srd in dataset.shuffle(buffer_size=10, seed=42).repeat(10).batch(16):
    print(srd)

# %% [markdown]
# ## Preprocessing using `tf.data`

# %% [markdown]
# ### Creating the split dataset

# %%
import cudf as cd
import dask_cudf as dcd
from sklearn.datasets import fetch_california_housing

dcd.from_cudf(
    cd.from_dataframe(
        fetch_california_housing(as_frame=True)["frame"], allow_copy=True
    ),
    npartitions=5,
).to_csv(str(DATA_DIR / "california_housing"), index=False)

# %% [markdown]
# ### Reading csv using `tf.data`

# %%
import tensorflow as tf
from tensorflow import data, io

# %%
N_INPUTS = 8


def parse_csv_line(line: str) -> tuple[tf.Tensor, tf.Tensor]:
    fields = io.decode_csv(
        line, record_defaults=[0.0] * N_INPUTS + [tf.constant([], dtype=tf.float32)]
    )

    return tf.stack(fields[:-1]), tf.stack(fields[-1:])


# %%
N_READERS, N_THREADS, SEED = 5, 5, 42

dataset = (
    (
        data.Dataset.list_files(
            str(DATA_DIR / "california_housing" / "*.part"), seed=SEED
        )
        .interleave(
            lambda f: data.TextLineDataset(f).skip(1),
            cycle_length=N_READERS,
            num_parallel_calls=N_THREADS,
        )
        .map(parse_csv_line, num_parallel_calls=N_THREADS)
    )
    .shuffle(buffer_size=10_000)
    .batch(32)
    .prefetch(data.AUTOTUNE)
)

# %%
for d in dataset.take(1):
    print(d)

# %% [markdown]
# ## TfRecord Format

# %%
import tensorflow as tf
from tensorflow import data
from tensorflow.io import TFRecordOptions, TFRecordWriter

# %%
with TFRecordWriter(str(TFR_DIR / "first.tfrecord")) as f:
    f.write(b"First Record: 01")
    f.write(b"First Record: 02")
    f.write(b"First Record: 03")

with TFRecordWriter(str(TFR_DIR / "second.tfrecord")) as f:
    f.write(b"Second Record: 01")
    f.write(b"Second Record: 02")
    f.write(b"Second Record: 03")

with TFRecordWriter(str(TFR_DIR / "third.tfrecord")) as f:
    f.write(b"Third Record: 01")
    f.write(b"Third Record: 02")
    f.write(b"Third Record: 03")

# %%
import glob

rec_ds = data.TFRecordDataset(
    glob.glob(str(TFR_DIR / "*.tfrecord")), num_parallel_reads=3
)

for r in rec_ds:
    print(r)

# %% [markdown]
# ## Protocol Buffers

# %%
import tensorflow as tf
from tensorflow import data
from tensorflow.io import TFRecordOptions, TFRecordWriter

# %% [raw]
# %%writefile protobufs/person.proto
# syntax = "proto3";
# message Person {
#     string name = 1;
#     int32 id = 2;
#     repeated string emails = 3;
# }

# %% [raw]
# !protoc --python_out=. --include_imports \
#     --descriptor_set_out=protobufs/person.desc \
#     protobufs/person.proto

# %%
from protobufs.person_pb2 import Person

person = Person(name="Al", id=22, emails=["al@alexandro.com"])
person.emails.append("john@richard.org")

# %%
print(person)
print(person.name)
print(person.id)
person.name = "Alexandro"
print(person.name)

# %%
print((serialized := person.SerializeToString()))

# %%
person2 = Person()
person2.ParseFromString(serialized) == len(serialized)

# %%
person2 == person

# %% [markdown]
# ### Decoding Custom Protobuf using Tensorflow Op

# %%
# tf.io.decode_proto?

# %%
(
    person_tf := tf.io.decode_proto(
        bytes=serialized,
        message_type="Person",
        field_names=["name", "id", "emails"],
        output_types=[tf.string, tf.int32, tf.string],
        descriptor_source=str(PROTO_DIR / "person.desc"),
    )
)

# %%
person_tf.values

# %% [markdown]
# ### TensorFlow Protobufs

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
from tensorflow.train import BytesList, Example, Feature, Features, Int64List

(
    person_example := Example(
        features=Features(
            feature={
                "name": Feature(bytes_list=BytesList(value=[b"Alejandro"])),
                "id": Feature(int64_list=Int64List(value=[1])),
                "emails": Feature(
                    bytes_list=BytesList(
                        value=[b"al@alejandro.com", b"bal@balkrishna.org"]
                    )
                ),
            }
        )
    )
)

# %%
tfr_options = {"compression_type": "GZIP"}

with TFRecordWriter(
    str(DATA_DIR / "tfrecords" / "person_example.tfrecord"),
    options=TFRecordOptions(**tfr_options),
) as f:
    f.write(person_example.SerializeToString())

del person_example

# %%
# Example?

# %%
from tensorflow.io import FixedLenFeature, VarLenFeature

person_ds = data.TFRecordDataset(
    str(DATA_DIR / "tfrecords" / "person_example.tfrecord.gz"), **tfr_options
).map(
    lambda ex: tf.io.parse_example(
        ex,
        features={
            "name": VarLenFeature(dtype=tf.string),
            "id": FixedLenFeature(shape=(), dtype=tf.int64),
            "emails": VarLenFeature(dtype=tf.string),
        },
    ),
    num_parallel_calls=data.AUTOTUNE,
)

# %%
for p in person_ds:
    print(tf.sparse.to_dense(p["name"]))
    print(p["id"])
    print(p["emails"].values)

# %% [markdown]
# ### Serializing Images and Tensors

# %%
from matplotlib import pyplot as plt
from sklearn.datasets import load_sample_images

img = load_sample_images()["images"][0]
plt.imshow(img)
plt.title("Original Image")
plt.axis(False);

# %%
# Only necessary if image is not already a .jpeg
tf.io.encode_jpeg(img)

# %%
img_example = Example(
    features=Features(
        feature={
            "image": Feature(
                bytes_list=BytesList(value=[tf.io.encode_jpeg(img).numpy()])
            )
        }
    )
)

# %%
with TFRecordWriter(
    str(TFR_DIR / "image_example.tfrecord.gz"), options=TFRecordOptions(**tfr_options)
) as f:
    f.write(img_example.SerializeToString())

del img_example

# %%
img_ds = data.TFRecordDataset(
    str(TFR_DIR / "image_example.tfrecord.gz"), **tfr_options
).map(
    lambda ex: tf.io.decode_jpeg(
        tf.io.parse_single_example(
            ex, features={"image": VarLenFeature(dtype=tf.string)}
        )["image"].values[0]
    ),
    num_parallel_calls=data.AUTOTUNE,
)

# %%
for i in img_ds:
    plt.imshow(i)
    plt.title("Decoded Image")
    plt.axis(False)

# %% [markdown]
# ## `SequenceExample` Protobuf

# %% [markdown]
# ```proto
# syntax = "proto3";
#
# message FeatureList { repeated Feature feature = 1; };
# message FeatureLists { map<string, FeatureList> feature_list = 1; };
# message SequenceExample {
#     Features context = 1;
#     FeatureLists feature_lists = 2;
# };
# ```

# %%
from typing import Iterable

from tensorflow.train import FeatureList, FeatureLists, SequenceExample

context = Features(
    feature={
        "author_id": Feature(int64_list=Int64List(value=[123])),
        "title": Feature(bytes_list=BytesList(value=[b"A", b"Desert", b"Place", b"."])),
        "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25])),
    }
)

content = [
    ["When", "shall", "we", "three", "meet", "again", "?"],
    ["In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"],
]
comments = [
    ["When", "the", "hurlyburly", "'s", "done", "."],
    ["When", "the", "battle", "'s", "lost", "and", "won", "."],
]


def words_to_feature(words: Iterable[str]) -> Feature:
    return Feature(
        bytes_list=BytesList(value=[bytes(word, encoding="utf8") for word in words])
    )


(
    seq_example := SequenceExample(
        context=context,
        feature_lists=FeatureLists(
            feature_list={
                "content": FeatureList(
                    feature=[words_to_feature(words) for words in content]
                ),
                "comments": FeatureList(
                    feature=[words_to_feature(words) for words in comments]
                ),
            }
        ),
    )
)

# %%
with TFRecordWriter(
    str(TFR_DIR / "sequence_example.tfrecord.gz"),
    options=TFRecordOptions(**tfr_options),
) as f:
    f.write(seq_example.SerializeToString())

del seq_example

# %%
seq_ds = data.TFRecordDataset(
    str(TFR_DIR / "sequence_example.tfrecord.gz"), **tfr_options
).map(
    lambda ex: tf.io.parse_single_sequence_example(
        ex,
        context_features={
            "author_id": FixedLenFeature(shape=(), dtype=tf.int64),
            "title": VarLenFeature(dtype=tf.string),
            "pub_date": FixedLenFeature(shape=(3,), dtype=tf.int64),
        },
        sequence_features={
            "content": VarLenFeature(dtype=tf.string),
            "comments": VarLenFeature(dtype=tf.string),
        },
    )
)

# %%
for parsed_context, s in seq_ds:
    parsed_sequences = {k: tf.RaggedTensor.from_sparse(v) for k, v in s.items()}

# %%
parsed_sequences

# %% [markdown]
# ## Keras Preprocessing Layers

# %%
import tensorflow as tf
from tensorflow import data, keras
from tensorflow.keras import layers

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

del housing, X_train_full, y_train_full
gc.collect()

# %%
norm = layers.Normalization()
model = keras.Sequential([norm, layers.Dense(1)])

# %%
