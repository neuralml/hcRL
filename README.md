# hcRL

Code and data for the paper:  
**"Hippocampal networks support reinforcement learning in partially observable environments"**

---

## 1. System Requirements

- **Operating System**: Tested on Ubuntu 20.04  
- **Python version**: Python 3.9  
- **Hardware**: GPU (CUDA-compatible), GeForce RTX 2080 Ti.

### Software Dependencies

| Package     | Version    |
|-------------|------------|
| Python      | 3.9        |
| PyTorch     | ≥ 1.12.0   |
| Hydra       | ≥ 1.2.0    |
| wandb       | ≥ 0.18.7   |
| numpy       | ≥ 1.24     |
| gym         | ≥ 0.26.2   |
| omegaconf   | ≥ 2.2.2    |
| minigrid    | ≥ 2.3.1    |


---

## 2. Installation Guide

### Step 1: Clone the repository

```sh
git clone https://github.com/neuralml/hcRL.git
cd hcRL
```

### Step 2: Create and activate the Conda environment
```sh
conda env create -n hcRL
conda activate hcRL
```

## 3. Instructions
To run a full training session with your own configuration:
```
python train.py agent=dqn seed=1 experiment=exp use_wandb=true
```
To train hcDRQN run 
```
python train.py agent=drqn seed=1 experiment=exp use_wandb=true
```

Results are logged and tracked via Weights & Biases, if enabled.

## 4. License

This project is licensed under the MIT License. See LICENSE for details.

