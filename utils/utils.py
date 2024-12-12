import argparse
import logging
import os
import pprint
import random
import time

import numpy as np
import torch
import yaml

from config import load_config


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO, save_dir=None):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        fh = logging.FileHandler(os.path.join(save_dir, f'log.txt'),
                                 mode='w')  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def load_state(path, model, optimizer=None, key="state_dict"):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        print("path = ", path)
        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    print(
                        "caution: size-mismatch key: {} size: {} -> {}".format(
                            k, v.shape, v_dst.shape
                        )
                    )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        best_metric = checkpoint["best_acc"]
        if checkpoint.get("epoch", None) is not None:
            last_iter = checkpoint["epoch"]
            print(
                "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                    path, last_iter
                )
            )
            return best_metric, last_iter, None
        else:
            last_iter = checkpoint["iters"]
            print(
                "=> also loaded optimizer from checkpoint '{}' (iters {})".format(
                    path, last_iter
                )
            )
            return best_metric, None, last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))


def init():
    parser = argparse.ArgumentParser(description="run")
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)

    # set random seed
    seed = cfg['seed']
    if seed is not None:
        print("set random seed to {}".format(seed))
        set_random_seed(seed)

    # calculate path
    cfg["exp_path"] = os.path.dirname(args.config)
    snapshot_dir = cfg.get("snapshot_dir", None)
    if snapshot_dir is None:
        snapshot_dir = 'checkpoints_' + '_'.join(args.config.split('.')[0].split('_')[1:])
        cfg["snapshot_dir"] = snapshot_dir
    cfg["save_path"] = os.path.join(cfg["exp_path"], snapshot_dir)

    logger = init_log("global", logging.INFO, save_dir=cfg["save_path"])
    logger.propagate = 0

    # save config.yaml
    if not os.path.exists(cfg["save_path"]):
        os.makedirs(cfg["save_path"])
    cfg_dump_path = f'{cfg["save_path"]}/config.yaml'
    with open(cfg_dump_path, mode='w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
        print(f'config.yaml dumped at [{cfg_dump_path}]')

    pprint.pprint(cfg)
    return cfg, logger
