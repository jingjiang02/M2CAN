# Multi-source Multi-modal Domain Adaptation

> **Multi-source Multi-modal Domain Adaptation**<br>
> TODO<br>

<!-- [YouTube](https://www.youtube.com/watch?v=c55rRO-Aqac&ab_channel=JaeminNa)<br> -->
> **Abstract:** *TODO*

## Table of Contents

- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Getting Started](#getting-started)
- [Citation](#Citation)

## Introduction

TODO

## Requirements

- Linux
- Python >= 3.7
- PyTorch == 1.10.1
- opencv-python == 4.8.0.76
- CUDA (must be a version supported by the pytorch version)

## Getting Started

### data prepare

1. aesthetics
    1. [AVA](https://github.com/imfing/ava_downloader)
    2. [RPCD](https://github.com/mediatechnologycenter/aestheval)
    3. [PCCD](https://github.com/ivclab/DeepPhotoCritic-ICCV17)
2. sentiment
    1. [TumEmo](https://github.com/YangXiaocui1215/MVAN)
    2. [T4SA](http://www.t4sa.it/)
    3. [Yelp](https://github.com/PreferredAI/vista-net)

The train-test subset split is in the directory [split](./split).

### train

Take our **msmm** for example:

```bash
cd msmm/scripts/
bash train.sh $config
```

- `$config` denotes the config file, for example **config_AVA.yaml**

### test

```bash
cd msmm/scripts/
bash test.sh $ckpt_dir
```

- `$ckpt_dir` denotes the checkpoint directory, for example **checkpoints_AVA**

## Citation

If you use this code in your research, please cite:

```bibtex
TODO
```
