from __future__ import annotations

import copy
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
import optax.contrib
import optax.tree
from flax import linen as nn
from flax.training import train_state
from jax._src.numpy import reductions, ufuncs

from models import optimizer_index, nn_index

EPS = 1e-8


@chex.dataclass
class IdentityScalerState:
    """State that mimics a scaler with a persistent scale field."""
    count: jnp.ndarray  # step count (int32)
    scale: jnp.ndarray  # always 1 (float32 by default)


def scale_by_identity(dtype=jnp.float32) -> optax.GradientTransformation:
    """
    Returns a no-op gradient transformation whose state has a `.scale` field that
    always equals 1.0. Useful as a stand-in for scalers (e.g., reduce_lr_on_plateau)
    when you want the same API surface but no learning-rate scaling.

    Example:
        tx = optax.chain(
            scale_by_identity(),          # exposes state.scale == 1
            optax.scale(-1.0)             # normal optimizer step scaling
        )
    """

    def init_fn(params):
        del params
        return IdentityScalerState(
            count=jnp.array(0, dtype=jnp.int32),
            scale=jnp.array(1.0, dtype=dtype),
        )

    # Accept **extra so you can pass metrics or other kwargs without errors,
    # mirroring contrib transforms that accept `value=...`, etc.
    @jax.jit
    def update_fn(updates, state: IdentityScalerState, params=None, **extra: Any):
        del params, extra
        new_state = IdentityScalerState(
            count=state.count + jnp.array(1, dtype=jnp.int32),
            scale=state.scale,  # stays 1.0 forever
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


class ScalerTrainState(train_state.TrainState):
    @jax.jit
    def apply_gradients_with_scaler(self, *, grads, scaler_state):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params
        )
        updates = optax.tree.scale(scaler_state.scale, updates)

        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )


def reset_optimizer_state(state, tx):
    state = state.replace(tx=tx)
    return state.replace(opt_state=tx.init(state.params))


def one_hot_smooth(labels, num_classes, smoothing=0.05):
    on_value = 1.0 - smoothing
    off_value = smoothing / (num_classes - 1)
    eye = jnp.eye(num_classes, dtype=jnp.float32)
    return eye[labels] * on_value + (1.0 - eye[labels]) * off_value


# Using optax as optimer
@jax.jit
def apply_model(state, images, labels, **kwargs):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        targets = nn.one_hot(labels, logits.shape[1])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def apply_model_smoothing(state, images, labels, smoothing=0.0, **kwargs):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        targets = one_hot_smooth(labels, logits.shape[1], smoothing=smoothing)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def apply_model_regularization(state, images, labels, regularization_coefficient=0.0, prev_task_state=None, **kwargs):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        targets = nn.one_hot(labels, logits.shape[1])
        unregularized_loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))
        vec_cur_params, _ = jax.flatten_util.ravel_pytree(params)
        vec_prev_params, _ = jax.flatten_util.ravel_pytree(prev_task_state.params)
        regularization = regularization_coefficient * vector_l2(vec_cur_params - vec_prev_params)
        loss = unregularized_loss + regularization
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def apply_model_smoothing_regularization(state, images, labels, smoothing=0.0, regularization_coefficient=0.0,
                                         prev_task_state=None, **kwargs):
    def loss_fn(params):
        logits = state.apply_fn(params, images)
        targets = one_hot_smooth(labels, logits.shape[1], smoothing=smoothing)
        unregularized_loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))
        vec_cur_params, _ = jax.flatten_util.ravel_pytree(params)
        vec_prev_params, _ = jax.flatten_util.ravel_pytree(prev_task_state.params)
        regularization = regularization_coefficient * vector_l2(vec_cur_params - vec_prev_params)
        loss = unregularized_loss + regularization
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def apply_model_no_grad(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        """cross-entropy loss function"""
        logits_model = state.apply_fn(params, images)
        one_hot = nn.one_hot(labels, logits_model.shape[1])  # keeping testing unsmoothed on purpose
        loss_model = jnp.mean(optax.softmax_cross_entropy(logits=logits_model, labels=one_hot))
        return loss_model, logits_model

    loss, logits = loss_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


@jax.jit
def update_model(state, grads):
    """gradient update model"""
    return state.apply_gradients(grads=grads)


def train_epoch(state, data_loader, scaler=None, scaler_state=None, smoothing=0.0, regularization_coefficient=0.0,
                prev_task_state=None):
    """Train for a single epoch.
    train_group_label: original labels of classes label in dataset
    random_classes_label: specific labels assigned from 0 to num_classes in classification, it could be random assigned
    """
    epoch_loss_arr, epoch_accuracy_arr = [], []
    model_norms_arr, gradient_norms_arr = [], []
    lr_scale_arr = []
    model_norms_arr.append(parameters_norm(state.params))
    if smoothing > 0.0:
        if regularization_coefficient > 0.0:
            model_apply_function = apply_model_smoothing_regularization
        else:
            model_apply_function = apply_model_smoothing
    else:
        if regularization_coefficient > 0.0:
            model_apply_function = apply_model_regularization
        else:
            model_apply_function = apply_model
    if scaler is None or scaler_state is None:
        scaler = scale_by_identity()
        scaler_state = scaler.init(state.params['params'])
    for batch_images, batch_labels in data_loader:
        grads, loss, accuracy = model_apply_function(state, batch_images, batch_labels, smoothing=smoothing,
                                                     regularization_coefficient=regularization_coefficient,
                                                     prev_task_state=prev_task_state)
        state = state.apply_gradients_with_scaler(grads=grads, scaler_state=scaler_state)
        epoch_loss_arr.append(loss)
        epoch_accuracy_arr.append(accuracy)
        model_norms_arr.append(parameters_norm(state.params))
        gradient_norms_arr.append(parameters_norm(grads))
        lr_scale_arr.append(scaler_state.scale)
    train_loss = jnp.mean(jnp.array(epoch_loss_arr))
    prev_scale = scaler_state.scale
    _, scaler_state = scaler.update(
        updates=state.params['params'], state=scaler_state, value=train_loss
    )
    if scaler_state.scale != prev_scale:
        print("Scaler updated, new scale:", scaler_state.scale.item())
    return state, scaler_state, epoch_loss_arr, epoch_accuracy_arr, model_norms_arr, gradient_norms_arr, lr_scale_arr


def test_model(trained_model_state, test_ds):
    """Test for trained model."""
    test_loss = []
    test_accuracy = []
    batch_sizes = []
    for batch in test_ds:
        # Average test accuracy
        test_images, test_labels = batch
        loss, accuracy = apply_model_no_grad(trained_model_state, test_images, test_labels)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
        batch_sizes.append(test_images.shape[0])
    batch_sizes = jnp.array(batch_sizes)
    mean_loss = jnp.sum(jnp.array(test_loss) * batch_sizes) / jnp.sum(batch_sizes)
    mean_accuracy = jnp.sum(jnp.array(test_accuracy) * batch_sizes) / jnp.sum(batch_sizes)
    return mean_loss, mean_accuracy


def reset_optimizer_and_scaler_between_tasks(model_state, params):
    optimizer = optimizer_index(params['optimizer'], params['learning_rate'],
                                momemtum=params['momentum__momentum'],
                                nesterov=params['momentum__nesterov'])
    model_state = reset_optimizer_state(model_state, optimizer)
    scaler = init_scaler(params)
    scaler_state = scaler.init(model_state.params['params'])
    return model_state, scaler, scaler_state


def init_scaler(params):
    if params['plateau']:
        scaler = optax.contrib.reduce_on_plateau(params['plateau__factor'], params['plateau__patience'],
                                                 params['plateau__tolerance'], accumulation_size=1)
    else:
        scaler = scale_by_identity()
    return scaler


def initialize_model(params):
    model_input_shape = (params['batch_size'], *params['input_size'])  # batch_size, image shape
    model = nn_index(params['nn_type'])
    optimizer = optimizer_index(params['optimizer'], params['learning_rate'],
                                momemtum=params['momentum__momentum'], nesterov=params['momentum__nesterov'])
    model_keys = jax.random.split(jax.random.PRNGKey(params['ini_seed']), 2)
    model_params = model.init(model_keys[0],
                              jax.random.normal(model_keys[1], model_input_shape))
    model_state = ScalerTrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)
    return model_state


def recursive_list_of_lists(shape_tuple):
    """create a list of lists with given shape"""
    if len(shape_tuple) == 1:
        return [[] for _ in range(shape_tuple[0])]
    else:
        return [recursive_list_of_lists(shape_tuple[1:]) for _ in range(shape_tuple[0])]


def update_losses_and_accuracy_of_each_task(params, model_state, train_ds_list, test_ds_list, histories, i_perm,
                                            verbose=True):
    for i_task in range(params['num_tasks']):
        train_loss, train_acc = test_model(model_state, train_ds_list[i_task])
        test_loss, test_acc = test_model(model_state, test_ds_list[i_task])
        cur_results = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        for key in histories.keys():
            histories[key][i_perm][i_task].append(cur_results[key])
    if verbose:
        for key in histories.keys():
            mean_metric = jnp.mean(jnp.asarray(histories[key][i_perm]), axis=0)[-1]
            print(f'Average across all tasks {key}: {mean_metric:.4f}')
    return histories


@jax.jit
def parameters_norm(params):
    vec, _ = jax.flatten_util.ravel_pytree(params)
    return jnp.linalg.norm(vec)


def trainstate_deepcopy(state: train_state.TrainState) -> train_state.TrainState:
    # Copy every JAX/NumPy array buffer; leave non-arrays (e.g., apply_fn) as-is.
    def _copy_leaf(x):
        if isinstance(x, (jnp.ndarray,)):
            return jnp.array(x, copy=True)  # new buffer on same device
        try:
            # optax states sometimes include small Python containers
            return copy.deepcopy(x)
        except Exception:
            return x

    return jax.tree_util.tree_map(_copy_leaf, state)


@jax.jit
def model_state_l2_distance(state_a, state_b):
    vec_a, _ = jax.flatten_util.ravel_pytree(state_a.params)
    vec_b, _ = jax.flatten_util.ravel_pytree(state_b.params)
    return jnp.linalg.norm(vec_a - vec_b)


def vector_l2(x):
    return reductions.sum(ufuncs.real(x * ufuncs.conj(x)), keepdims=False)
