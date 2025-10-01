from __future__ import annotations

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp


# CNN structure for binary classification
class CNN(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


# nonlinear multilayer neuro network for binary classification
class DNN(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


class Linear(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_outputs)(x)
        return x


# return neuro network based on nn_label
def nn_index(nn_type, num_outputs=2):
    if nn_type == 'cnn':
        nn_model = CNN(num_outputs)
    elif nn_type == 'dnn':
        nn_model = DNN(num_outputs)
    elif nn_type == 'linear':
        nn_model = Linear(num_outputs)
    else:
        raise ValueError('nn_type must be cnn, dnn, or linear')
    return nn_model


def optimizer_index(optimizer_type, lr, **kwargs):
    if optimizer_type == 'adam':
        optimizer = optax.adam(learning_rate=lr)
    elif optimizer_type == 'sgd':
        optimizer = optax.sgd(learning_rate=lr)
    elif optimizer_type == 'momentum':
        optimizer = optax.sgd(learning_rate=lr, momentum=kwargs['momentum'], nesterov=kwargs['nesterov'])
    else:
        raise ValueError('optimizer_type must be adam, sgd, or momentum')
    chain = optax.named_chain(("nan_guard", zero_nans_and_zero_large(1e3)),
                              ('solver', optimizer))
    return chain


def zero_nans_and_zero_large(threshold: float) -> optax.GradientTransformation:
    """Replace NaNs with 0 and zero out elements with |x| > threshold."""
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        def sanitize(x):
            x = jnp.where(jnp.isnan(x), 0.0, x)
            x = jnp.where(jnp.isposinf(x) | jnp.isneginf(x), 0.0, x)
            return jnp.where(jnp.abs(x) > threshold, 0.0, x)
        updates = jax.tree_util.tree_map(sanitize, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
