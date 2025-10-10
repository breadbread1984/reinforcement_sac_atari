# Introduction

this project implements Soft Actor-Critic (SAC) reinforcement learning on Atari games

# Usage

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## Training with customized deep learning model

```shell
python3 train_dm.py --device (cuda|cpu)
```

## Testing with customized deep learning model

```shell
python3 test_dm.py --ckpt <path/to/ckpt> --device (cuda|cpu)
```
