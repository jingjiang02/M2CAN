from transformers import BertTokenizer

from data_loader.common_loader import build_common_data_loader


def get_loader(cfg, source_combined=False, seed=0):
    cfg_dataset = cfg["dataset"]
    train_loaders, val_loaders = [], []

    tokenizer = BertTokenizer.from_pretrained(cfg["dataset"]["bert_model"])

    origin_batch_size = cfg["dataset"]["batch_size"]
    for _type in cfg_dataset["types"]:
        print(_type)

        # Handling source combined situations [currently only implementing two source domains]
        if source_combined:
            if _type != cfg_dataset["types"][-1]:
                cfg["dataset"]["batch_size"] = origin_batch_size // 2
            else:
                cfg["dataset"]["batch_size"] = origin_batch_size

        train_loader = build_common_data_loader("train", _type, cfg, tokenizer, seed=seed)
        val_loader = build_common_data_loader("val", _type, cfg, tokenizer, seed=seed)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    return train_loaders, val_loaders
