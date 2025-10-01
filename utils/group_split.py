import math
import random

import jax
import jax.numpy as jnp


# pick specific classes from all classes in the dataset and rearrange these labels into group [[C] for P]
def random_pick_into_groups(seed, num_total_classes=100, num_tasks=20, group_size=2):
    key = jax.random.PRNGKey(seed)  # PRNGKey to avoid same pick
    permutation = jax.random.permutation(key, jnp.arange(num_total_classes))
    numbers = permutation[:num_tasks * group_size]
    groups = numbers.reshape((num_tasks, group_size))
    return groups.tolist()


def split_cifar100_by_superclasses(num_classes_per_task=2, num_tasks=50, seed=None):
    if seed is not None:
        random.seed(seed)
    # CIFAR-100 has 20 superclasses, each with 5 fine labels
    coarse_to_fine = {
        i: list(range(5 * i, 5 * (i + 1))) for i in range(20)
    }
    # coarse_to_fine[0] -> [0,1,2,3,4], coarse_to_fine[1] -> [5,6,7,8,9], etc.
    all_coarse = list(coarse_to_fine.keys())
    random.shuffle(all_coarse)
    num_superclasses_per_group = math.ceil(num_tasks / 5)
    groups = [all_coarse[i * num_superclasses_per_group:(i + 1) * num_superclasses_per_group] for i in
              range(num_classes_per_task)]
    groups_fine = [[item for sublist in [coarse_to_fine[superclass] for superclass in group] for item in sublist] for
                   group in groups]
    [random.shuffle(group_fine) for group_fine in groups_fine]
    return [[groups_fine[class_i][task_i] for class_i in range(num_classes_per_task)] for task_i in range(num_tasks)]
