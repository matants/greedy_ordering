import os

import numpy as np


class FeatureDataset:
    """
    Re-iterable, reshuffling dataset built from precomputed features.
    Every __iter__() call yields exactly one epoch.
    """

    def __init__(self, npz_path, group_labels, target_labels,
                 batch_size=64, shuffle=True, seed=0, dtype=np.float32,
                 shuffle_labels=False, percentage_to_keep=1.0):
        data = np.load(npz_path)
        feats = data["feats"]  # shape [N, D]
        labels = data["labels"]  # shape [N]

        if percentage_to_keep < 1.0:
            feats_out, labels_out = [], []
            unique_labels = np.unique(labels)
            rng_for_label = np.random.default_rng(seed)
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                n_keep = max(1, int(len(indices) * percentage_to_keep))  # at least 1
                chosen = rng_for_label.choice(indices, size=n_keep, replace=False)
                feats_out.append(feats[chosen])
                labels_out.append(labels[chosen])
            feats = np.vstack(feats_out)
            labels = np.concatenate(labels_out)

        # Filter to the two (or more) task classes
        group_labels = np.asarray(group_labels, dtype=np.int32)
        mask = np.isin(labels, group_labels)
        self.X = feats[mask].astype(dtype, copy=False)
        y = labels[mask].astype(np.int32, copy=False)

        # Vectorized remap original -> target (e.g., [orig_a, orig_b] -> [0, 1])
        max_label = int(labels.max()) + 1
        lut = np.full(max_label, -1, dtype=np.int32)
        lut[group_labels] = np.asarray(target_labels, dtype=np.int32)
        self.y = lut[y]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)  # persisted across epochs
        self.shuffle_labels = bool(shuffle_labels)

    def __len__(self):
        # number of batches in one epoch (no drop_remainder)
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # one epoch
        idx = np.arange(len(self.X))

        if self.shuffle_labels:
            self.rng.shuffle(self.y)

        if self.shuffle:
            self.rng.shuffle(idx)  # different order each epoch

        # yield batches
        for s in range(0, len(idx), self.batch_size):
            ii = idx[s:s + self.batch_size]
            yield self.X[ii], self.y[ii]


def gen_embedded_ds_load(group_labels, target_random_labels, params, source_dir="features/",
                         generate_shuffled_labels_train=False, percentage_to_keep=1.0):
    train_npz = os.path.join(source_dir, "train.npz")
    test_npz = os.path.join(source_dir, "test.npz")
    if not os.path.exists(train_npz) or not os.path.exists(test_npz):
        raise FileNotFoundError("Feature files not found, generate them first. See README.md for instructions.")

    train_ds_list, test_ds_list = [], []
    shuffled_labels_train_ds_list = []
    for (g_lbls, tgt_lbls) in zip(group_labels, target_random_labels):
        # Training: reshuffles each epoch automatically
        train_ds = FeatureDataset(
            train_npz, g_lbls, tgt_lbls,
            batch_size=params['batch_size'],
            shuffle=True,
            seed=params['ini_seed'],
            percentage_to_keep=percentage_to_keep
        )
        # Test: deterministic order (no shuffle)
        test_ds = FeatureDataset(
            test_npz, g_lbls, tgt_lbls,
            batch_size=params['batch_size'],
            shuffle=False,
            seed=params['ini_seed'],
            percentage_to_keep=percentage_to_keep
        )
        train_ds_list.append(train_ds)
        test_ds_list.append(test_ds)

        if generate_shuffled_labels_train:
            train_ds_shuffled_labels = FeatureDataset(
                train_npz, g_lbls, tgt_lbls,
                batch_size=params['batch_size'],
                shuffle=True,
                seed=params['ini_seed'],
                shuffle_labels=True,
                percentage_to_keep=percentage_to_keep
            )
            shuffled_labels_train_ds_list.append(train_ds_shuffled_labels)

    return train_ds_list, test_ds_list, shuffled_labels_train_ds_list
