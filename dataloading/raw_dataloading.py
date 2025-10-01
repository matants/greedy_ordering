import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

AUTOTUNE = tf.data.AUTOTUNE


def make_label_table(group_label, target_label):
    """
    group_label: Python list/array of original CIFAR-100 class ids, e.g. [37, 12]
    target_label: Python list/array of desired ids for this task, e.g. [0, 1]
    """
    keys = tf.constant(group_label, dtype=tf.int64)
    vals = tf.constant(target_label, dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(-1, tf.int64)  # anything not in group -> -1
    )
    return table


def normalize_and_relabel(table):
    def _map(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        new_label = table.lookup(tf.cast(label, tf.int64))
        return image, tf.cast(new_label, tf.int32)

    return _map


def filter_by_labels_tensor(group_labels_tensor):
    def _filter(image, label):
        return tf.reduce_any(tf.equal(tf.cast(label, tf.int64), group_labels_tensor))

    return _filter


def build_ds_for_task(dataset, group_label, target_label, batch_size, shuffle_size, training=True,
                      shuffle_labels=False):
    # Convert group_label once to tensor (faster than capturing a Python list in the graph)
    g_tensor = tf.constant(group_label, dtype=tf.int64)
    table = make_label_table(group_label, target_label)

    ds = dataset.filter(filter_by_labels_tensor(g_tensor))
    ds = ds.map(normalize_and_relabel(table), num_parallel_calls=AUTOTUNE)

    if shuffle_labels:
        ds = shuffled_labels_ds(ds, shuffle_size)

    if training:
        ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return tfds.as_numpy(ds)


# Function to filter dataset based on label groups
def filter_by_labels(group_labels):
    # Define the labels you want to keep: group_labels
    def filter_fn(image, label):
        # Check if the label is in the provided group
        return tf.reduce_any(tf.equal(label, group_labels))

    return filter_fn


# Function to normalize and resize the dataset
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to range [0, 1]
    return image, label


# Resize image data to specific image size, not necessary
# def resize_image(image, label):
#   image = tf.image.resize(image, [image_size0, image_size1])  # image_size initialization in the run.py
#   return image, label


# regularization of train dataset
def train_ds_norm(train_ds, batch_size, shuffle_size):
    train_ds = train_ds.map(normalize_image)
    train_ds = train_ds.shuffle(shuffle_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = tfds.as_numpy(train_ds)
    return train_ds


def train_ds_norm_shuffled_labels(train_ds, batch_size, shuffle_size):
    train_ds = train_ds.map(normalize_image)
    images_ds = train_ds.map(lambda image, label: image)
    labels_ds = train_ds.map(lambda image, label: label).shuffle(shuffle_size)
    shuffled_train_ds = tf.data.Dataset.zip(images_ds, labels_ds)
    shuffled_train_ds = shuffled_train_ds.shuffle(shuffle_size).batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    shuffled_train_ds = tfds.as_numpy(shuffled_train_ds)
    return shuffled_train_ds


def shuffled_labels_ds(ds, shuffle_size):
    images_ds = ds.map(lambda image, label: image)
    labels_ds = ds.map(lambda image, label: label).shuffle(shuffle_size)
    shuffled_ds = tf.data.Dataset.zip(images_ds, labels_ds)
    return shuffled_ds


# dataset upload
def ds_upload(ds_dir, ds_type):
    (train_dataset, test_dataset), info = tfds.load(ds_type, split=['train', 'test'], as_supervised=True,
                                                    data_dir=ds_dir, with_info=True)
    return train_dataset, test_dataset


# Generation of dataloader and split into several groups
def gen_ds_load(group_labels, const_params, train_dataset, test_dataset, target_random_labels=None,
                generate_shuffled_labels_train=False):
    batch_size = const_params['batch_size']
    shuffle_size = const_params['shuffle_size']

    # If none given, default to identity mapping [0..group_size-1]
    if target_random_labels is None:
        target_random_labels = [list(range(len(g))) for g in group_labels]

    train_ds_list, test_ds_list = [], []
    shuffled_labels_train_ds_list = []
    for group_label, target_label in zip(group_labels, target_random_labels):
        train_ds = build_ds_for_task(train_dataset, group_label, target_label,
                                     batch_size, shuffle_size, training=True)
        test_ds = build_ds_for_task(test_dataset, group_label, target_label,
                                    batch_size, shuffle_size, training=False)
        train_ds_list.append(train_ds)
        test_ds_list.append(test_ds)
        if generate_shuffled_labels_train:
            train_ds_shuffled_labels = build_ds_for_task(train_dataset, group_label, target_label,
                                                         batch_size, shuffle_size, training=True, shuffle_labels=True)
            shuffled_labels_train_ds_list.append(train_ds_shuffled_labels)
    return train_ds_list, test_ds_list, shuffled_labels_train_ds_list
