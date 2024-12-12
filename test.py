import argparse
import logging
import os

from common_modules.utils import eval_on
from config import load_config
from data_loader.builder import get_loader
from msmm.models.model_msmm import MsmmModel
from utils.utils import (
    init_log,
    load_state,
    set_random_seed
)

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--config", type=str, default="config.yaml")
# ckpt file path
parser.add_argument("--pth_file", default=None, type=str)
# Model name
parser.add_argument("--model_name", default=None, type=str)

logger = init_log("global", logging.INFO)
logger.propagate = 0
digits = 4


def main():
    global args, cfg
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg['seed']
    if seed is not None:
        logger.info("set random seed to {}".format(seed))
        set_random_seed(seed)

    # Create network
    if args.model_name is None:
        raise Exception("model name is None")
    elif args.model_name.lower() == 'msmm':
        model = MsmmModel(cfg)
    else:
        raise Exception("unsupported model name")
    logger.info(f'starting evaluate model [{args.model_name}]!')

    model.cuda()

    _, val_loaders = get_loader(cfg, seed=seed)

    pth_file = args.pth_file
    if not os.path.exists(pth_file):
        raise Exception(f'no file in [{pth_file}]')
    elif os.path.isdir(pth_file):
        pth_file = f'{args.pth_file}/ckpt_best.pth'
    else:
        pass

    pth_file = os.path.abspath(pth_file)
    logger.info(f"Resume model from: '{pth_file}'")
    _, best_epoch, _ = load_state(
        pth_file, model, optimizer=None, key="model_state"
    )
    logger.info(f'best_epoch={best_epoch}')

    # Validation
    eval_on(
        cfg,
        model,
        val_loaders,
        -1,
        logger,
        epoch=best_epoch,
        best_epoch=best_epoch,
        save_best=False
    )


if __name__ == "__main__":
    main()
