from utils.gpu_utils import setup_gpu

setup_gpu(min_free_gb=12)

from typing import Dict, Any
from collections import defaultdict
import os
import argparse
import numpy as np
import time
import json
import jax.numpy as jnp

from utils.argparse_utils import int_or_literal_factory, str2bool, parse_list, load_config_file, merge_params, \
    save_config_file
from dataloading.pretrained_dataloading import gen_embedded_ds_load
from dataloading.raw_dataloading import ds_upload, gen_ds_load
from utils.group_split import random_pick_into_groups, split_cifar100_by_superclasses
from learning_utils import train_epoch, reset_optimizer_and_scaler_between_tasks, initialize_model, \
    recursive_list_of_lists, update_losses_and_accuracy_of_each_task, test_model, EPS, trainstate_deepcopy, \
    model_state_l2_distance
from utils.json_utils import array_to_jsonable
from utils.model_saving import save_checkpoint, load_checkpoint


def build_base_defaults() -> Dict[str, Any]:
    return {
        # data
        "ds_type": "cifar100",  # 'cifar100'
        # "is_pretrained": False,  # True, False
        # "data_dir": "tfds",  # 'tfds', 'features_cifar100', 'features_cifar10', 'features_cifar100_split'
        # "input_size": [32, 32, 3],  # [32, 32, 3] for cifar, [64] for pretrained features
        "is_pretrained": True,  # True, False
        "data_dir": "features_cifar100",  # 'tfds', 'features_cifar100', 'features_cifar10', 'features_cifar100_split'
        "input_size": [64],  # [32, 32, 3] for cifar, [64] for pretrained features
        "num_all_classes": 100,  # total classes

        # tasks
        "num_tasks": 50,
        "num_output_classes": 2,
        "create_new_tasks": True,
        "tasks_origin_file_template": "experiments/cifar100_e2e_tfds_cnn_sgd_plateau_lr_0.1_epochs_100_batch_64_tasks_10_classes_2_index_{num_index}_{i_pick}_baseline.json",
        # "tasks_origin_file_template": "experiments/cifar100_pretrained_linear_greedy_md_P50_C2_index{num_index}__{i_pick}_task_selection_results.json",
        "splitting": "cifar100_superclasses",  # 'cifar100_superclasses', 'random'

        # model
        "nn_type": "linear",  # 'cnn', 'dnn', 'linear'
        # "nn_type": "cnn",  # 'cnn', 'dnn', 'linear'

        # training
        "learning_rate": 0.01,
        # "learning_rate": 0.1,
        "num_epochs": 40,
        "num_training_iterations": 'num_tasks',  # 'num_tasks' or an integer
        "batch_size": 64,
        "shuffle_size": 1000,
        "optimizer": "sgd",  # 'sgd', 'adam', 'momentum'
        "momentum__momentum": 0.9,
        "momentum__nesterov": True,
        "plateau": True,
        "plateau__factor": 0.5,
        "plateau__patience": 4,
        "plateau__tolerance": 1e-4,
        "smoothing": 0.05,
        "regularization": 5,

        # parameters for experiment setting
        "ordering": "greedy_mr_no_shuffle",
        # 'baseline', 'random', 'greedy_mr_no_shuffle', 'greedy_mr_shuffle', 'greedy_md'
        "data_size_for_rule_estimation": 1.0,
        "repetitions": False,
        "num_index": 70,  # job index
        "num_pick": 2,  # number of ways to pick/group tasks
        "random__num_perm": 4,
        "ini_seed": 0,

        # "load_model_path": "model_checkpoints/cifar100_pretrained_features_cifar100_split_linear_sgd_plateau_lr_0.01_epochs_40_batch_64_tasks_5_classes_2_index_999_0_baseline_perm_0.npy"  # str or None
        "load_model_path": None  # str or None
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run experiment with file defaults and CLI overrides",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    # meta/config controls
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON/YAML config file to preload defaults.")
    p.add_argument("--save-config", type=str, default=None,
                   help="Optional path to write the final merged config (JSON or YAML).")
    # We add all real experiment args with SUPPRESS defaults so only provided flags override.
    # Data
    p.add_argument("--ds_type", type=str, default=argparse.SUPPRESS)
    p.add_argument("--is_pretrained", type=str2bool, default=argparse.SUPPRESS)
    p.add_argument("--data_dir", type=str, default=argparse.SUPPRESS)
    p.add_argument("--input_size", type=parse_list, default=argparse.SUPPRESS)
    p.add_argument("--num_all_classes", type=int, default=argparse.SUPPRESS)

    # Tasks
    p.add_argument("--num_tasks", type=int, default=argparse.SUPPRESS)
    p.add_argument("--num_output_classes", type=int, default=argparse.SUPPRESS)
    p.add_argument("--create_new_tasks", type=str2bool, default=argparse.SUPPRESS)
    p.add_argument("--tasks_origin_file_template", type=str, default=argparse.SUPPRESS)
    p.add_argument("--splitting", type=str, default=argparse.SUPPRESS)

    # Model
    p.add_argument("--nn_type", type=str, default=argparse.SUPPRESS)

    # Training
    p.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS)
    p.add_argument("--num_epochs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--num_training_iterations", type=int_or_literal_factory("num_tasks"), default=argparse.SUPPRESS)
    p.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--shuffle_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--optimizer", type=str, default=argparse.SUPPRESS)
    p.add_argument("--momentum__momentum", type=float, default=argparse.SUPPRESS)
    p.add_argument("--momentum__nesterov", type=str2bool, default=argparse.SUPPRESS)
    p.add_argument("--plateau", type=str2bool, default=argparse.SUPPRESS)
    p.add_argument("--plateau__factor", type=float, default=argparse.SUPPRESS)
    p.add_argument("--plateau__patience", type=int, default=argparse.SUPPRESS)
    p.add_argument("--plateau__tolerance", type=float, default=argparse.SUPPRESS)
    p.add_argument("--smoothing", type=float, default=argparse.SUPPRESS)
    p.add_argument("--regularization", type=float, default=argparse.SUPPRESS)

    # Experiment settings
    p.add_argument("--ordering", type=str, default=argparse.SUPPRESS)
    p.add_argument("--data_size_for_rule_estimation", type=float, default=argparse.SUPPRESS)
    p.add_argument("--repetitions", type=str2bool, default=argparse.SUPPRESS)
    p.add_argument("--num_index", type=int, default=argparse.SUPPRESS)
    p.add_argument("--num_pick", type=int, default=argparse.SUPPRESS)
    p.add_argument("--random__num_perm", type=int, default=argparse.SUPPRESS)
    p.add_argument("--ini_seed", type=int, default=argparse.SUPPRESS)

    return p


###############################################################################################################

def run(params: Dict[str, Any]):
    # generate implicit settings
    regularization_coefficient = params['regularization']
    if params['ordering'] == 'baseline':
        num_training_iterations = 1
        regularization_coefficient = 0
    elif params['num_training_iterations'] == 'num_tasks':
        num_training_iterations = params['num_tasks']
    else:
        num_training_iterations = int(params['num_training_iterations'])
    if num_training_iterations > params['num_tasks'] and not params['repetitions']:
        raise ValueError("num_training_iterations must be <= num_tasks if repetitions is False")
    num_perms = params['random__num_perm'] if params['ordering'] == 'random' else 1
    if params['data_size_for_rule_estimation'] < 1.0 and not params['is_pretrained']:
        raise NotImplementedError("data_size_for_rule_estimation < 1.0 is not implemented for non-pretrained datasets")
    if params['data_size_for_rule_estimation'] < 1.0 and params['ordering'] == 'greedy_md':
        raise NotImplementedError("data_size_for_rule_estimation < 1.0 is currently not implemented for ordering 'greedy_md'")


    # start loop
    for i_pick in range(params['num_pick']):
        print(f"num_index: {params['num_index']}")
        print(f"num_pick: {i_pick}")
        experiment_random_seed = params['ini_seed'] + int(params['num_index'] * params['num_pick']) + i_pick
        file_name_parts = [params['ds_type'],
                           'pretrained' if params['is_pretrained'] else 'e2e',
                           params['data_dir'],
                           params['nn_type'],
                           params['optimizer'],
                           'plateau' if params['plateau'] else 'constant',
                           'lr', str(params['learning_rate']),
                           'regularization', str(params['regularization']),
                           'epochs', str(params['num_epochs']),
                           'batch', str(params['batch_size']),
                           'tasks', str(params['num_tasks']),
                           'classes', str(params['num_output_classes']),
                           'index', str(params['num_index']), str(i_pick),
                           params['ordering']
                           ]
        if params['data_size_for_rule_estimation'] != 1.0 and params['ordering'].startswith('greedy'):
            file_name_parts.extend(['partial', str(params['data_size_for_rule_estimation'])])
        if params['repetitions']:
            file_name_parts.extend(['repetitions', str(params['num_training_iterations'])])

        """
        main function running
        """
        start_time = time.time()
        # tasks building
        if params['create_new_tasks']:
            if params['splitting'] == 'random':
                group_labels = random_pick_into_groups(experiment_random_seed,
                                                       num_total_classes=params['num_all_classes'],
                                                       num_tasks=params['num_tasks'],
                                                       group_size=params[
                                                           'num_output_classes'])  # original label of class in ds
            elif params['splitting'] == 'cifar100_superclasses':
                group_labels = split_cifar100_by_superclasses(num_classes_per_task=params['num_output_classes'],
                                                              num_tasks=params['num_tasks'],
                                                              seed=experiment_random_seed)
            else:
                raise ValueError("params['splitting'] must be 'random' or 'cifar100_superclasses'")
            target_labels = [list(range(params['num_output_classes'])) for _ in range(params['num_tasks'])]
        else:
            with open(params['tasks_origin_file_template'].format(num_index=params['num_index'], i_pick=i_pick),
                      "r") as json_f:
                tasks_dict = json.load(json_f)
            group_labels = tasks_dict['group_labels']
            target_labels = tasks_dict['target_labels']

        print("group_labels:", group_labels)
        print("target_labels:", target_labels)

        # data loading

        if not params['is_pretrained']:
            train_ds, test_ds = ds_upload(params['data_dir'], params['ds_type'])
            train_ds_list, test_ds_list, shuffled_labels_train_ds_list = gen_ds_load(
                group_labels, params, train_ds, test_ds, target_labels,
                generate_shuffled_labels_train=(params['ordering'] == 'greedy_mr_shuffle'))
        else:
            train_ds_list, test_ds_list, shuffled_labels_train_ds_list = gen_embedded_ds_load(
                group_labels, target_labels, params, source_dir=params['data_dir'],
                generate_shuffled_labels_train=(params['ordering'] == 'greedy_mr_shuffle'))
            if params['data_size_for_rule_estimation'] < 1.0:
                train_ds_list_for_rule_estimation, _, shuffled_labels_train_ds_list_for_rule_estimation = gen_embedded_ds_load(
                    group_labels, target_labels, params, source_dir=params['data_dir'],
                    generate_shuffled_labels_train=(params['ordering'] == 'greedy_mr_shuffle'),
                    percentage_to_keep=params['data_size_for_rule_estimation'])

        # generate unified datasets for baseline training
        if params['ordering'] == 'baseline':
            group_labels_unified = [np.asarray(group_labels).flatten().tolist()]
            target_labels_unified = [np.asarray(target_labels).flatten().tolist()]
            if not params['is_pretrained']:
                train_ds_list_unified, test_ds_list_unified, _ = gen_ds_load(group_labels_unified, params, train_ds,
                                                                             test_ds,
                                                                             target_labels_unified)
            else:
                train_ds_list_unified, test_ds_list_unified, _ = gen_embedded_ds_load(group_labels_unified,
                                                                                      target_labels_unified, params,
                                                                                      source_dir=params['data_dir'])
            assert len(train_ds_list_unified) == 1

        print("data ready")

        histories = {
            k: recursive_list_of_lists((num_perms, params['num_tasks'])) for k in
            ['train_acc', 'test_acc', 'train_loss', 'test_loss']
        }  # [num_perms, num_tasks, num_measurements (num_tasks + 1)]
        epochs_acc, epochs_loss, model_norms, grad_norms, lr_scales = (
            recursive_list_of_lists((num_perms, num_training_iterations)) for _ in range(5)
        )  # [num_perms, num_training_iterations, num_epochs, batches_per_epoch (or +1)]
        orderings = []
        residuals = []

        for i_perm in range(num_perms):
            perm_file_name_parts = file_name_parts + ["perm", f"{i_perm}"]
            available_tasks = np.arange(params['num_tasks'])
            selected_ordering = []
            residuals_for_rule_calculation = []

            if params['ordering'] == 'random':
                perm_seed = num_perms * experiment_random_seed + i_perm
                np.random.seed(perm_seed)

            # initialize model
            model_state = initialize_model(params)
            if params["load_model_path"] is not None:
                model_state = load_checkpoint(params["load_model_path"], model_state)
            selected_task = None
            for training_iteration in range(num_training_iterations):
                # update losses and accuracy of each task
                histories = update_losses_and_accuracy_of_each_task(params, model_state, train_ds_list, test_ds_list,
                                                                    histories, i_perm)

                # select task for learning
                if selected_task is not None:
                    available_tasks_excluding_previous = available_tasks[available_tasks != selected_task]
                else:
                    available_tasks_excluding_previous = available_tasks

                if params['ordering'] == 'baseline':
                    selected_task = 0
                elif params['ordering'] == 'random':
                    selected_task = int(np.random.choice(available_tasks_excluding_previous).item())
                elif params['ordering'] == 'greedy_mr_shuffle':
                    available_tasks_losses_normalized = {}
                    for task in available_tasks_excluding_previous:
                        if params['data_size_for_rule_estimation'] == 1.0:
                            task_train_loss = histories['train_loss'][i_perm][task][-1]
                            task_train_loss_shuffled_labels, _ = test_model(model_state,
                                                                            shuffled_labels_train_ds_list[task])
                        else:
                            task_train_loss, task_train_acc = test_model(model_state,
                                                                         train_ds_list_for_rule_estimation[task])
                            task_train_loss_shuffled_labels, _ = test_model(model_state,
                                                                            shuffled_labels_train_ds_list_for_rule_estimation[task])
                        available_tasks_losses_normalized[int(task)] = (
                                task_train_loss / (task_train_loss_shuffled_labels + EPS)).item()
                    residuals_for_rule_calculation.append(available_tasks_losses_normalized)
                    selected_task = max(available_tasks_losses_normalized, key=available_tasks_losses_normalized.get)
                elif params['ordering'] == 'greedy_mr_no_shuffle':
                    available_tasks_losses = {}
                    for task in available_tasks_excluding_previous:
                        if params['data_size_for_rule_estimation'] == 1.0:
                            task_train_loss = histories['train_loss'][i_perm][task][-1]
                        else:
                            task_train_loss, task_train_acc = test_model(model_state,
                                                                         train_ds_list_for_rule_estimation[task])
                        available_tasks_losses[int(task)] = task_train_loss.item()
                    residuals_for_rule_calculation.append(available_tasks_losses)
                    selected_task = max(available_tasks_losses, key=available_tasks_losses.get)
                elif params['ordering'] == 'greedy_md':
                    available_tasks_distances = {}
                    model_state, scaler, scaler_state = reset_optimizer_and_scaler_between_tasks(model_state, params)
                    (potential_epochs_loss, potential_epochs_acc, potential_model_norms, potential_grad_norms,
                     potential_lr_scales) = [defaultdict(list) for _ in range(5)]
                    maximal_distance = -np.inf
                    best_model = None
                    for task in available_tasks_excluding_previous:
                        task = int(task)
                        new_state = trainstate_deepcopy(model_state)
                        new_state, scaler, scaler_state = reset_optimizer_and_scaler_between_tasks(new_state, params)
                        for epoch in range(params['num_epochs']):
                            (new_state, scaler_state,
                             train_loss, train_accuracy, model_norms_epoch, grad_norms_epoch,
                             lr_scales_epoch) = train_epoch(
                                new_state,
                                train_ds_list[task],
                                scaler=scaler,
                                scaler_state=scaler_state,
                                smoothing=params['smoothing'],
                                regularization_coefficient=regularization_coefficient,
                                prev_task_state=model_state
                            )
                            train_loss = jnp.asarray(train_loss)
                            train_accuracy = jnp.asarray(train_accuracy)
                            potential_epochs_loss[task].append(train_loss)
                            potential_epochs_acc[task].append(train_accuracy)
                            potential_model_norms[task].append(model_norms_epoch)
                            potential_grad_norms[task].append(grad_norms_epoch)
                            potential_lr_scales[task].append(lr_scales_epoch)
                            print(
                                f'Potential task: {task}, epoch: {epoch:03d}, train loss: {jnp.mean(train_loss).item():.4f},'
                                f' train accuracy: {jnp.mean(train_accuracy).item():.4f}, model norm: {model_norms_epoch[-1].item():.4f},')
                        distance = model_state_l2_distance(new_state, model_state).item()
                        available_tasks_distances[task] = distance
                        if distance > maximal_distance:
                            maximal_distance = distance
                            best_model = new_state

                    residuals_for_rule_calculation.append(available_tasks_distances)
                    selected_task = max(available_tasks_distances, key=available_tasks_distances.get)

                    model_state = best_model
                    epochs_loss[i_perm][training_iteration] = potential_epochs_loss[selected_task]
                    epochs_acc[i_perm][training_iteration] = potential_epochs_acc[selected_task]
                    model_norms[i_perm][training_iteration] = potential_model_norms[selected_task]
                    grad_norms[i_perm][training_iteration] = potential_grad_norms[selected_task]
                    lr_scales[i_perm][training_iteration] = potential_lr_scales[selected_task]

                else:
                    raise NotImplementedError

                if not params['repetitions']:
                    available_tasks = available_tasks[available_tasks != selected_task]
                selected_ordering.append(selected_task)

                # train model
                if params['ordering'] != 'greedy_md':
                    if params['ordering'] != 'baseline':
                        training_ds = train_ds_list[selected_task]
                    else:
                        training_ds = train_ds_list_unified[0]
                    model_state, scaler, scaler_state = reset_optimizer_and_scaler_between_tasks(model_state, params)
                    prev_task_state = trainstate_deepcopy(model_state)

                    for epoch in range(params['num_epochs']):
                        (model_state, scaler_state,
                         train_loss, train_accuracy, model_norms_epoch, grad_norms_epoch,
                         lr_scales_epoch) = train_epoch(
                            model_state,
                            training_ds,
                            scaler=scaler,
                            scaler_state=scaler_state,
                            smoothing=params['smoothing'],
                            regularization_coefficient=regularization_coefficient,
                            prev_task_state=prev_task_state
                        )
                        train_loss = jnp.asarray(train_loss)
                        train_accuracy = jnp.asarray(train_accuracy)
                        epochs_loss[i_perm][training_iteration].append(train_loss)
                        epochs_acc[i_perm][training_iteration].append(train_accuracy)
                        model_norms[i_perm][training_iteration].append(model_norms_epoch)
                        grad_norms[i_perm][training_iteration].append(grad_norms_epoch)
                        lr_scales[i_perm][training_iteration].append(lr_scales_epoch)
                        print(
                            f'epoch: {epoch:03d}, train loss: {jnp.mean(train_loss).item():.4f},'
                            f' train accuracy: {jnp.mean(train_accuracy).item():.4f}, model norm: {model_norms_epoch[-1].item():.4f},')

                print(len(selected_ordering), 'tasks done:', selected_ordering)

            orderings.append(selected_ordering)
            residuals.append(residuals_for_rule_calculation)
            # update losses and accuracy of each task at the end of the run
            histories = update_losses_and_accuracy_of_each_task(params, model_state, train_ds_list, test_ds_list,
                                                                histories,
                                                                i_perm)
            # save model
            model_dir = 'model_checkpoints'
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir, '_'.join(perm_file_name_parts) + '.npy')
            save_checkpoint(model_state, model_filename)
            print(f"saved model to {model_filename}")

            print("finished ordering run")

        calc_duration = time.time() - start_time
        os.makedirs('experiments', exist_ok=True)
        json_file_name = os.path.join('experiments', '_'.join(file_name_parts) + ".json")
        dict_to_dump = {
            'calc_duration': calc_duration,
            'epochs_acc': array_to_jsonable(epochs_acc),
            'epochs_loss': array_to_jsonable(epochs_loss),
            'model_norms': array_to_jsonable(model_norms),
            'grad_norms': array_to_jsonable(grad_norms),
            'lr_scales': array_to_jsonable(lr_scales),
            'orderings': orderings,
            'residuals': residuals,
            'group_labels': group_labels,
            'target_labels': target_labels,
            'params': params
        }
        dict_to_dump.update({k: array_to_jsonable(histories[k]) for k in histories})
        with open(json_file_name, "w") as json_f:
            json.dump(dict_to_dump, json_f)
        print(f"saved to {json_file_name}")


def main(raw_args=None):
    base_defaults = build_base_defaults()
    # 1) parse config path first
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre.add_argument("--save-config", type=str, default=None)
    known, _ = pre.parse_known_args()

    file_defaults = load_config_file(known.config) if known.config else {}
    parser = build_parser()
    # 2) parse CLI overrides (only provided flags will appear in Namespace)
    args = parser.parse_args(raw_args)
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k not in ("config", "save_config")}
    final_params = merge_params(base_defaults, file_defaults, cli_overrides)

    if known.save_config:
        save_config_file(known.save_config, final_params)

    run(final_params)


if __name__ == '__main__':
    main()
