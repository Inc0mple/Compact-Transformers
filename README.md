# 50.035 Computer Vision Project Research Track: Compact Vision Transformers for Classification

## Report | Slides | [Original Repository](https://github.com/SHI-Labs/Compact-Transformers)

## Group Members
Bryan Tan (1004318) <br>
Christy Lau Jin Yun (1005330) <br>
Hung Chia-Yu (1005330) <br>
Mandis Loh (1005297) <br>
Jared Lim (1005200) <br>


## Setup Environment

- Clone this repository
- Create a new conda environment, activate it, and do `pip install -r requirements.txt`
- Download [cifar10 dataset](https://www.dropbox.com/scl/fi/oj44dqj4mlmzemntj32nv/cifar10.7z?rlkey=o8ncggr8u2gjilyaqof68a1ic&dl=0) and [cifar100 dataset](https://www.dropbox.com/scl/fi/jro158zl6h70uhhumava8/cifar100.7z?rlkey=3346pw04zhjev4t245omofui7&dl=0) and place them in `data/cifar10` and `data/cifar100` respectively.

## Important Files

1. `train.py` - main training script
2. `src/cct.py`
3. `src/utils/transformers.py`
4. `src/utils/tokenizer.py`
5. `configs/custom/cifar10_no_amp.yml` and `configs/custom/cifar100_no_amp.yml`
6. `macs_counter.ipynb`

## Training

To train a model, run `python train.py <path to dataset> -c <path to config file> --model <model name> --epochs <number of epochs> --output <path to output folder> --experiment <name of experiment> --log-wandb`

### Instructions
- add `--log-wandb` to log to wandb. Remove if you just want to experiment and not log to wandb.
- Models (both the ones by the original authors and our custom ones) are defined in `src/cct.py` and `src/cvt.py`. Edit the models there.
- Configurations for datasets are defined in `configs/`. Define your own config file if you want to use a different set of parameters.
- `--experiment` is the name of the experiment, which will be the name of the run used to save the model and log to wandb.
- `--output` saves the model and result to the specified folder in your local machine.

### Example Usage 

```bash
python train.py data/cifar10 -c configs/datasets/cifar10.yml --model cct_2_3x2_32  --epochs 10 --output result --experiment trial_cct_2_3x2_32 --log-wandb

python train.py data/cifar10 -c configs/datasets/cifar10.yml --model cct_2_3x2_32  --epochs 300 --output result --experiment full_cct_2_3x2_32 --log-wandb

python train.py data/cifar10 -c configs/datasets/cifar10.yml --model cct_custom_model --epochs 10 --output result --experiment example_trial_run_custom_model_here --log-wandb
```

## Visualisation/Evaluation
- Download the 8GB [results file here](https://drive.google.com/file/d/1PfQ78J8LvG0ptWzOiBanSNpcqyj8WeZa/view?usp=sharing) and unzip `result_final.7z`. Place the `result_final` folder in the root directory of this repository. This folder contains the results and checkpoints of all the models we trained.
- 



## Instructions




```
