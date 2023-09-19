# 50.035 Computer Vision Project

## [Original Repository](https://github.com/SHI-Labs/Compact-Transformers)


## Installation

- Clone this repository
- Create a new conda environment, activate it, and do `pip install -r requirements.txt`
- Download [cifar10 dataset](https://www.dropbox.com/scl/fi/oj44dqj4mlmzemntj32nv/cifar10.7z?rlkey=o8ncggr8u2gjilyaqof68a1ic&dl=0) and place it in the `data/cifar10` folder.

## Instructions
- add `--log-wandb` to log to wandb. Remove if you just want to experiment and not log to wandb.
- Models (both original and custom) are defined in `src/cct.py` and `src/cvt.py`. Edit the models there.
- Configurations for datasets are defined in `configs/`. Define your own config file if you want to use a different set of parameters.
- `--experiment` is the name of the experiment, which will be the name of the run used to save the model and log to wandb.
- `--output` saves the model and result to the specified folder in your local machine.


## Example Usage 

```bash
python train.py data/cifar10 -c configs/datasets/cifar10.yml --model cct_2_3x2_32  --epochs 10 --output result --experiment trial_cct_2_3x2_32 --log-wandb

python train.py data/cifar10 -c configs/datasets/cifar10.yml --model cct_2_3x2_32  --epochs 300 --output result --experiment full_cct_2_3x2_32 --log-wandb

python train.py data/cifar10 -c configs/datasets/cifar10.yml --model cct_custom_model --epochs 10 --output result --experiment example_trial_run_custom_model_here --log-wandb

```
