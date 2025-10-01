# Are Greedy Task Orderings Better Than Random in Continual Linear Regression?

This repository is the official implementation of classification experiments for [Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](URL_HERE). 

---

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Data generation
To create precomputed features for linear probing, run the script `dataloading/cifar100_resnet20_feats.py`.
Make sure to run it twice, once for train and once for test.
For using the data, specify the correct data location in the argument `data_dir` in `main.py`.
Similarly, it is possible to train a model on half of the samples of each $\texttt{CIFAR-100}$ class, and build training tasks from the other half. See `dataloading/cifar100_resnet20_feats_splits.py`.

##  Quick Start

### A) Default run (precomputed features, linear head, SGD)
```bash
python main.py
```

### B) Raw CIFAR-100 via TFDS (CNN)
```bash
python main.py \
  --is_pretrained false \
  --data_dir tfds \
  --input_size "[32,32,3]" \
  --nn_type cnn
```
## Configuration Options

You can set options in a **config file** (`--config`) and/or **override via CLI flags**. Only flags you provide are overridden; everything else comes from internal defaults + config file.

### Key Flags

**Data**
- `--ds_type {cifar100}`
- `--is_pretrained {true|false}` (default: `true`)
- `--data_dir {features_cifar100|features_cifar10|features_cifar100_split|tfds}`
- `--input_size "[64]"` (features) or `"[32,32,3]"` (images)
- `--num_all_classes INT` (default: `100`)

**Tasks**
- `--num_tasks INT` (default: `50`)
- `--num_output_classes INT` (default: `2`)
- `--create_new_tasks {true|false}`
- `--tasks_origin_file_template "experiments/...{num_index}_{i_pick}....json"`
- `--splitting {random|cifar100_superclasses}`

**Model**
- `--nn_type {linear|dnn|cnn}`

**Training**
- `--learning_rate FLOAT` (default: `0.01`)
- `--num_epochs INT` (default: `40`)
- `--num_training_iterations {INT|num_tasks}`
- `--batch_size INT` (default: `64`)
- `--shuffle_size INT`
- `--optimizer {sgd|adam|momentum}`
- `--momentum__momentum FLOAT` (e.g., `0.9`)
- `--momentum__nesterov {true|false}`
- `--plateau {true|false}` and `--plateau__factor FLOAT` / `--plateau__patience INT` / `--plateau__tolerance FLOAT`
- `--smoothing FLOAT` (label smoothing)
- `--regularization FLOAT`

**Experiment Control**
- `--ordering {baseline|random|greedy_mr_no_shuffle|greedy_mr_shuffle|greedy_md}`
- `--data_size_for_rule_estimation FLOAT` in `(0,1]` (only with `--is_pretrained true`, **not** with `greedy_md`)
- `--repetitions {true|false}`
- `--num_index INT` (job index), `--num_pick INT` (number of groupings), `--random__num_perm INT` (permutations for random)
- `--ini_seed INT`

**Checkpoints & Meta**
- `--load_model_path PATH` (resume/init model)
- `--config PATH` (load), `--save-config PATH` (write merged config)

## Outputs

- **Checkpoints** → `model_checkpoints/…npy`  
  Pattern (simplified):  
  `{ds_type}_{pretrained|e2e}_{data_dir}_{nn_type}_{optimizer}_{plateau|constant}_lr_{LR}_regularization_{REG}_epochs_{E}_batch_{B}_tasks_{T}_classes_{C}_index_{IDX}_{PICK}_{ordering}_perm_{PERM}.npy`

- **Metrics & metadata** → `experiments/…json`  
  Contains training/testing loss & accuracy traces, per-iteration selections (`orderings`), diagnostics (`residuals`), labels, and the resolved `params`.


---

This repository includes adaptations of code from: [Z. Li and N. Hiratani. Optimal task order for continual learning of multiple tasks. In Forty-second International Conference on Machine Learning, 2025.](https://github.com/ziyan-li-code/optimal-learn-order)

---


## Citation
If this work contributes to your research, please consider citing:
```
@inproceedings{tsipory2025greedy,
  title={Are Greedy Task Orderings Better Than Random in Continual Linear Regression?},
  author={Tsipory, Matan and Levinstein, Ran and Evron, Itay and Kong, Mark and Needell, Deanna and Soudry, Daniel},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
