import yaml

DEFAULTS = {
    'dataset': {
        'batch_size': 8,
        'bert_model': '/home/jjiang/experiments/m2can/pretrain_model/bert-base-uncased/',
        'train': {
            'TumEmo': {
                'data_list': '/home/jjiang/experiments/m2can/split/sentiment/TumEmo/train.json',
                'data_root': '/home/jjiang/datasets/TumEmo',
                'n_sample': 15000
            },
            'Twitter': {
                'data_list': '/home/jjiang/experiments/m2can/split/sentiment/Twitter/train.json',
                'data_root': '/home/jjiang/datasets/Twitter',
                'n_sample': 15000
            },
            'Yelp': {
                'data_list': '/home/jjiang/experiments/m2can/split/sentiment/Yelp/train.json',
                'data_root': '/home/jjiang/datasets/Yelp',
                'n_sample': 15000
            },
            'PCCD': {
                'data_list': '/home/jjiang/experiments/m2can/split/aesthetic/PCCD/train.json',
                'data_root': '/home/jjiang/datasets//PCCD/PCCD/images/full',
                'n_sample': 3388
            },
            'AVA': {
                'data_list': '/home/jjiang/experiments/m2can/split/aesthetic/AVA/train.json',
                'data_root': '/home/jjiang/datasets/AVA/images4AVA/images',
                'n_sample': 3388
            },
            'RPCD': {
                'data_list': '/home/jjiang/experiments/m2can/split/aesthetic/RPCD/train.json',
                'data_root': '/home/jjiang/datasets/RPCD/RPCD',
                'n_sample': 3388
            },
            'text_encode': {
                'max_length': 200
            },
            'transforms_strong': {
                'mean': [0.485, 0.456, 0.406],
                'rand_apply': {
                    'color_jitter': {
                        'brightness': 0.4,
                        'contrast': 0.4,
                        'hue': 0.1,
                        'saturation': 0.4
                    },
                    'p': 0.8
                },
                'rand_flip': 0.5,
                'rand_gray_scale': 0.2,
                'rand_resize_crop': 224,
                'std': [0.229, 0.224, 0.225]
            },
            'transforms_weak': {
                'center_crop': 224,
                'mean': [0.485, 0.456, 0.406],
                'resize': 256,
                'std': [0.229, 0.224, 0.225]
            }
        },
        'val': {
            'TumEmo': {
                'data_list': '/home/jjiang/experiments/m2can/split/sentiment/TumEmo/val.json',
                'data_root': '/home/jjiang/datasets/TumEmo',
                'n_sample': 1500
            },
            'Twitter': {
                'data_list': '/home/jjiang/experiments/m2can/split/sentiment/Twitter/val.json',
                'data_root': '/home/jjiang/datasets/Twitter',
                'n_sample': 1500
            },
            'Yelp': {
                'data_list': '/home/jjiang/experiments/m2can/split/sentiment/Yelp/val.json',
                'data_root': '/home/jjiang/datasets/Yelp',
                'n_sample': 1500
            },
            'PCCD': {
                'data_list': '/home/jjiang/experiments/m2can/split/aesthetic/PCCD/val.json',
                'data_root': '/home/jjiang/datasets//PCCD/PCCD/images/full',
                'n_sample': 847
            },
            'AVA': {
                'data_list': '/home/jjiang/experiments/m2can/split/aesthetic/AVA/val.json',
                'data_root': '/home/jjiang/datasets/AVA/images4AVA/images',
                'n_sample': 847
            },
            'RPCD': {
                'data_list': '/home/jjiang/experiments/m2can/split/aesthetic/RPCD/val.json',
                'data_root': '/home/jjiang/datasets/RPCD/RPCD',
                'n_sample': 847
            },
            'text_encode': {
                'max_length': 200
            },
            'transforms_strong': {
                'mean': [0.485, 0.456, 0.406],
                'rand_apply': {
                    'color_jitter': {
                        'brightness': 0.4,
                        'contrast': 0.4,
                        'hue': 0.1,
                        'saturation': 0.4
                    },
                    'p': 0.8
                },
                'rand_flip': 0.5,
                'rand_gray_scale': 0.2,
                'rand_resize_crop': 224,
                'std': [0.229, 0.224, 0.225]
            },
            'transforms_weak': {
                'center_crop': 224,
                'mean': [0.485, 0.456, 0.406],
                'resize': 256,
                'std': [0.229, 0.224, 0.225]
            }
        },
        'workers': 8
    },
    'trainer': {
        'epochs': 10,
        'eval_mode': {
            'epoch': 1
        },
        'log_iter': 200,
        'eval_on': True,
        'lr_scheduler': {
            'kwargs': {
                'power': 0.99,
                'lowest_lr': 1e-6
            },
            'mode': 'poly'
        },
        'optimizer': {
            'backbone_lr': 2e-05,
            'classifier_lr': 0.0005,
            'extractor_lr': 2e-05,
            'kwargs': {
                'betas': '0.9 0.999',
                'eps': '1e-8',
                'lr': 0.001
            },
            'mlb_lr': 0.0005,
            'mmdiscriminator1_lr': 0.0002,
            'mmdiscriminator2_lr': 0.0003,
            'type': 'adam'
        }
    },
    'criterion': {
        'cross_modal_contrastive_loss_rate': 0.5,
        'cross_domain_contrastive_loss_rate': 0.2,
        'mm_D_loss_rate': 0.05
    },
    'net': {
        'ema_decay': 0.99,
        'num_classes': 3,
        'start_rate': 0.3,
        'update_rate': 0.3
    },
    'seed': 123456,
    'hyper_para': {
        'rho': 0.05,
        'eps': 0.8,
        'temperature': 2.0
    }
}

ignore_key = {
    'bert_model', 'data_list', 'data_root'
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
            else:
                if k in ignore_key:
                    dst[k] = v
        else:
            dst[k] = v


def load_config(config_file, defaults=None):
    if defaults is None:
        defaults = DEFAULTS
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    return config
