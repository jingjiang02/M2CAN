import logging
from math import cos, pi

import torch.optim as optim


def get_optimizer(parms, cfg_optim):
    """
    Get the optimizer
    """
    optim_type = cfg_optim["type"]
    optim_kwargs = dict(cfg_optim["kwargs"])
    optim_kwargs["eps"] = float(optim_kwargs["eps"])
    optim_kwargs['betas'] = (float(optim_kwargs['betas'].split(" ")[0]), float(optim_kwargs['betas'].split(" ")[1]))
    print("optim_kwargs = ", optim_kwargs)
    if optim_type == "SGD":
        optim_kwargs.pop('betas')
        optim_kwargs.pop('eps')
        optimizer = optim.SGD(parms, **optim_kwargs)
    elif optim_type == "adam":
        optimizer = optim.Adam(parms, **optim_kwargs)
    elif optim_type == "adamw":
        optimizer = optim.AdamW(parms, **optim_kwargs)
    elif optim_type == "adadelta":
        optim_kwargs.pop('betas')
        optimizer = optim.Adadelta(parms, **optim_kwargs)
    else:
        optimizer = None

    assert optimizer is not None, "optimizer type should is SGD, adam or adamw"

    return optimizer


def get_scheduler(cfg_trainer, len_data, optimizer, start_epoch=0, use_iteration=False):
    epochs = (
        cfg_trainer["epochs"] if not use_iteration else 1
    )  # if use_iteration = True, only one epoch be use
    lr_mode = cfg_trainer["lr_scheduler"]["mode"]
    lr_args = cfg_trainer["lr_scheduler"]["kwargs"]
    lr_scheduler = LRScheduler(
        lr_mode, lr_args, len_data, optimizer, epochs, start_epoch
    )
    return lr_scheduler


class LRScheduler(object):
    def __init__(self, mode, lr_args, data_size, optimizer, num_epochs, start_epochs):
        super(LRScheduler, self).__init__()
        logger = logging.getLogger("global")

        assert mode in ["multistep", "poly", "cosine"]
        self.mode = mode
        self.optimizer = optimizer
        self.data_size = data_size

        # start_epochs = start_epochs - 1
        self.cur_iter = start_epochs * data_size
        self.max_iter = num_epochs * data_size

        # set learning rate
        self.base_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.cur_lr = [lr for lr in self.base_lr]

        self.lowest_lr = lr_args.get("lowest_lr", 0.00000001)
        # poly kwargs
        # TODO
        if mode == "poly":
            self.power = lr_args["power"] if lr_args.get("power", False) else 0.9
            logger.info("The kwargs for lr scheduler: {}".format(self.power))
        if mode == "multistep":
            default_mist = list(range(0, num_epochs, num_epochs // 3))[1:]
            self.milestones = (
                lr_args["multistep"]
                if lr_args.get("multistep", False)
                else default_mist
            )
            logger.info("The kwargs for lr scheduler: {}".format(self.milestones))
        if mode == "cosine":
            self.targetlr = lr_args["targetlr"]
            logger.info("The kwargs for lr scheduler: {}".format(self.targetlr))

    def step(self):
        self._step()
        self.update_lr()
        self.cur_iter += 1

    def _step(self):
        # 增加最低学习率的设置
        if self.mode == "multistep":
            epoch = self.cur_iter // self.data_size
            power = sum([1 for s in self.milestones if s <= epoch])
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * pow(0.1, power)
                try:
                    adj_lr = max(adj_lr, self.lowest_lr)
                except TypeError:
                    adj_lr = self.lowest_lr
                self.cur_lr[i] = adj_lr
        elif self.mode == "poly":
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * (
                        (1 - float(self.cur_iter) / self.max_iter) ** (self.power)
                )
                # print(f"self.cur_iter = {self.cur_iter}  self.max_iter = {self.max_iter}  self.power = {self.power}  adj_lr = {adj_lr}  lr = {lr}")
                try:
                    adj_lr = max(adj_lr, self.lowest_lr)
                except TypeError:
                    adj_lr = self.lowest_lr
                self.cur_lr[i] = adj_lr
        elif self.mode == "cosine":
            for i, lr in enumerate(self.base_lr):
                adj_lr = (
                        self.targetlr
                        + (lr - self.targetlr)
                        * (1 + cos(pi * self.cur_iter / self.max_iter))
                        / 2
                )
                try:
                    adj_lr = max(adj_lr, self.lowest_lr)
                except TypeError:
                    adj_lr = self.lowest_lr
                self.cur_lr[i] = adj_lr
        else:
            raise NotImplementedError

    def get_lr(self):
        return self.cur_lr

    def update_lr(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.cur_lr):
            param_group["lr"] = lr
