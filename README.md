# hcRL
Code and data for paper 'Hippocampal networks support reinforcement learning in partially observable environments'.

## Requirements
This repository depends on the following frameworks:

- **[WeightAndBiases](https://www.wandb.com/)**: a tool for tracking and visualizing machine learning experiments.
- **[Hydra](https://hydra.cc/)**: a framework for managing complex applications, with support for configuration, logging, and more.

Create a Conda environment and install dependencies
```sh
conda env create -n hcRL
conda activate hcRL
```

## Instructions
To train hcDQN run
```
python test.py agent=dqn seed=1 experiment=exp use_wandb=true
```
To train hcDRQN run 
```
python test.py agent=drqn seed=1 experiment=exp use_wandb=true
```

