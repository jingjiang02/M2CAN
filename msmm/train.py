import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

import time

import torch
from tqdm import tqdm

from common_modules.mcc import MinimumClassConfusionLoss
from data_loader.builder import get_loader
from msmm.models.model_msmm import MsmmModel
from common_modules.modules import DomainDiscriminator
from common_modules.sdat import ConditionalDomainAdversarialLoss
from utils.lr_helper import get_scheduler, get_optimizer
from common_modules.sam import SAM
from utils.utils import (
    AverageMeter,
    init
)
from common_modules.utils import eval_on, get_batch


def main():
    # step1 init
    global cfg
    cfg, logger = init()
    seed = cfg['seed']

    # step2 Create network
    model = MsmmModel(cfg)
    modules_classifier = [classifier for classifier in model.classifiers]
    modules_classifier += [model.image_cls, model.text_cls]
    modules_backbone = [model.bert_model, model.model_image]
    modules_extractor = [model.text_feature_extractor, model.image_encoder]
    modules_mlb = [model.mlb]

    model.cuda()

    # step3 get loaders
    train_loaders, val_loaders = get_loader(cfg, seed=seed)

    # step4 Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]

    params_list = []
    for module in modules_backbone:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["backbone_lr"]))  # lr=0.00002, eps=1e-8
    for module in modules_extractor:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["extractor_lr"]))  # lr=0.0005, betas=(0.9, 0.999)
    for module in modules_mlb:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["mlb_lr"]))  # lr=0.0005, betas=(0.9, 0.999)
    for module in modules_classifier:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["classifier_lr"]))  # lr=0.0005, betas=(0.9, 0.999)

    # Optimizer hyperparameters
    lr = cfg['trainer']['optimizer']['kwargs']['lr']
    rho = cfg['hyper_para']['rho']

    base_optimizer = get_optimizer(params_list, cfg_optim)
    optimizer = SAM(params_list, base_optimizer, rho=rho, adaptive=False,
                    lr=lr)

    # Create MMDiscriminator
    num_classes = int(cfg['net']['num_classes'])
    mmdiscriminator1 = DomainDiscriminator(dim=64 * num_classes)
    mmdiscriminator2 = DomainDiscriminator(dim=64 * num_classes)
    mmdiscriminator3 = DomainDiscriminator(dim=64 * num_classes)
    params_list = []
    params_list.append(
        dict(params=mmdiscriminator1.parameters(),
             lr=cfg_optim["mmdiscriminator1_lr"]))  # lr=0.0002, betas=(0.9, 0.999)
    params_list.append(
        dict(params=mmdiscriminator2.parameters(),
             lr=cfg_optim["mmdiscriminator2_lr"]))  # lr=0.0002, betas=(0.9, 0.999)
    params_list.append(
        dict(params=mmdiscriminator3.parameters(),
             lr=cfg_optim["mmdiscriminator2_lr"]))  # lr=0.0002, betas=(0.9, 0.999)

    optim_kwargs = dict(cfg_optim["kwargs"])
    optim_kwargs["eps"] = float(optim_kwargs["eps"])
    optim_kwargs['betas'] = (float(optim_kwargs['betas'].split(" ")[0]), float(optim_kwargs['betas'].split(" ")[1]))
    print("optim_kwargs = ", optim_kwargs)
    ad_optimizer = get_optimizer(params_list, cfg_optim)

    # hyperparameters of MCC&SDAT
    randomized = False
    randomized_dim = 1024
    # 设置比例的！
    eps = cfg['hyper_para']['eps']
    domain_advs = torch.nn.ModuleList().cuda()
    domain_advs.append(ConditionalDomainAdversarialLoss(
        mmdiscriminator1, entropy_conditioning=False,
        num_classes=num_classes, features_dim=64 * num_classes, randomized=randomized,
        randomized_dim=randomized_dim, eps=eps
    ).cuda())
    domain_advs.append(ConditionalDomainAdversarialLoss(
        mmdiscriminator2, entropy_conditioning=False,
        num_classes=num_classes, features_dim=64 * num_classes, randomized=randomized,
        randomized_dim=randomized_dim, eps=eps
    ).cuda())
    domain_advs.append(ConditionalDomainAdversarialLoss(
        mmdiscriminator3, entropy_conditioning=False,
        num_classes=num_classes, features_dim=64 * num_classes, randomized=randomized,
        randomized_dim=randomized_dim, eps=eps
    ).cuda())

    # mcc_loss
    temperature = cfg['hyper_para']['temperature']
    mcc_loss = MinimumClassConfusionLoss(temperature=temperature)

    # Create MMDiscriminator end
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loaders[-1]), optimizer
    )

    ad_lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loaders[-1]), ad_optimizer
    )

    best_acc = 0
    best_epoch = 0
    start_rate = cfg['net']['start_rate']
    update_rate = cfg['net']['update_rate']
    # step5 Start to train model
    for epoch in tqdm(range(cfg_trainer["epochs"])):
        # Training

        if epoch == 0:
            train_loaders_for_pretrain = train_loaders[:2]
        else:
            target_name = cfg["dataset"]["types"][-1]
            cfg["dataset"]["train"][target_name]["data_list"] = model.pseudo_path
            train_loaders_new, _ = get_loader(cfg, seed=seed)
            train_loaders_for_pretrain = train_loaders[:2] + [train_loaders_new[-1]]
        train_pretrain(
            model,
            domain_advs,
            mcc_loss,
            optimizer,
            ad_optimizer,
            lr_scheduler,
            ad_lr_scheduler,
            train_loaders_for_pretrain,
            logger,
            epoch,
        )
        # Generate pseudo labels
        pseudo_rate = min(start_rate + (epoch * update_rate), 1.0)
        model.validate_pseudo(train_loaders[-1], pseudo_rate)

        # Validation
        if cfg_trainer["eval_on"] and cfg_trainer["eval_mode"].get("epoch", None) is not None:
            if epoch % cfg_trainer["eval_mode"]["epoch"] == 0:
                acc = eval_on(
                    cfg,
                    model,
                    val_loaders,
                    best_acc,
                    logger,
                    epoch=epoch,
                    best_epoch=best_epoch
                )
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch


def train_pretrain(
        model,
        domain_advs,
        mcc_loss,
        optimizer,
        ad_optimizer,
        lr_scheduler,
        ad_lr_scheduler,
        train_loaders,
        logger,
        epoch,
):
    model.train()
    domain_advs.train()

    # para
    mm_D_loss_rate = float(cfg["criterion"]["mm_D_loss_rate"])
    mcc_loss_weight = mm_D_loss_rate

    log_iter = cfg["trainer"]["log_iter"]

    loader_iters = []
    loader_length = len(train_loaders[-1])
    for loader in train_loaders:
        loader_iters.append(iter(loader))

    mcc_loss_meter = AverageMeter(log_iter)
    cross_modal_contrastive_loss_meter = AverageMeter(log_iter)
    cross_domain_contrastive_loss_meter = AverageMeter(log_iter)
    mm_D_loss_meter = AverageMeter(log_iter)
    task_loss_meter = AverageMeter(log_iter)
    total_loss_meter = AverageMeter(log_iter)
    batch_times = AverageMeter(log_iter)
    learning_rates = AverageMeter(log_iter)

    for step in range(loader_length):
        batch_start = time.time()

        i_iter = epoch * loader_length + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])

        train_batch = get_batch(loader_iters)

        optimizer.zero_grad()
        ad_optimizer.zero_grad()

        cross_modal_contrastive_loss, cross_domain_contrastive_loss, [_,
                                                                      preds], task_loss = model(
            train_batch)

        # mcc loss
        if len(preds) == 3:
            mcc_loss_value = mcc_loss_weight * mcc_loss(preds[-1])
        else:
            mcc_loss_value = torch.tensor(0.0).cuda()

        loss = task_loss + cross_modal_contrastive_loss + cross_domain_contrastive_loss + mcc_loss_value
        loss.backward()

        optimizer.first_step(zero_grad=True)

        cross_modal_contrastive_loss, cross_domain_contrastive_loss, [mm_features,
                                                                      preds], task_loss = model(
            train_batch)

        # cross-domain adversarial feature alignment (CDAFA)
        # s1 and s2
        mm_D_loss = domain_advs[0](preds[0], mm_features[0], preds[1], mm_features[1])
        if len(preds) == 3:
            # s1 and t
            mm_D_loss += domain_advs[1](preds[0], mm_features[0], preds[-1], mm_features[-1])
            # s2 and t
            mm_D_loss += domain_advs[2](preds[1], mm_features[1], preds[-1], mm_features[-1])

            mm_D_loss /= 3

        mm_D_loss *= mm_D_loss_rate

        # mcc loss
        if len(preds) == 3:
            mcc_loss_value = mcc_loss_weight * mcc_loss(preds[-1])
        else:
            mcc_loss_value = torch.tensor(0.0).cuda()

        loss = task_loss + cross_modal_contrastive_loss + cross_domain_contrastive_loss + mm_D_loss + mcc_loss_value

        loss.backward()
        # Update parameters of domain classifier
        ad_optimizer.step()
        # Update parameters (Sharpness-Aware update)
        optimizer.second_step(zero_grad=True)

        lr_scheduler.step()
        ad_lr_scheduler.step()

        mcc_loss_meter.update(mcc_loss_value.item())
        cross_modal_contrastive_loss_meter.update(cross_modal_contrastive_loss.item())
        cross_domain_contrastive_loss_meter.update(cross_domain_contrastive_loss.item())
        mm_D_loss_meter.update(mm_D_loss.item())
        task_loss_meter.update(task_loss.item())
        total_loss_meter.update(loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % log_iter == 0:
            logger.info(
                # "[{}] "
                "Iter [{}/{}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "total_loss_meter {total_loss_meter.val:.4f} ({total_loss_meter.avg:.4f})\t"
                "cross_modal_contrastive_loss_meter {cross_modal_contrastive_loss_meter.val:.4f} ({cross_modal_contrastive_loss_meter.avg:.4f})\t"
                "cross_domain_contrastive_loss_meter {cross_domain_contrastive_loss_meter.val:.4f} ({cross_domain_contrastive_loss_meter.avg:.4f})\t"
                "mm_D_loss_meter {mm_D_loss_meter.val:.4f} ({mm_D_loss_meter.avg:.4f})\t"
                "mcc_loss_meter {mcc_loss_meter.val:.4f} ({mcc_loss_meter.avg:.4f})\t"
                "task_loss_meter {task_loss_meter.val:.4f} ({task_loss_meter.avg:.4f})\t"
                "LR {lr.val:.7f}".format(
                    # cfg["dataset"]["n_sup"],
                    i_iter,
                    cfg["trainer"]["epochs"] * loader_length,
                    batch_time=batch_times,
                    total_loss_meter=total_loss_meter,
                    cross_modal_contrastive_loss_meter=cross_modal_contrastive_loss_meter,
                    cross_domain_contrastive_loss_meter=cross_domain_contrastive_loss_meter,
                    mm_D_loss_meter=mm_D_loss_meter,
                    mcc_loss_meter=mcc_loss_meter,
                    task_loss_meter=task_loss_meter,
                    lr=learning_rates,
                )
            )


if __name__ == "__main__":
    main()
