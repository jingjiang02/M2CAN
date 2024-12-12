import copy
import json
import os
import os.path
import random
import re
from typing import Any, List, Optional, Dict, Union, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

from .base import BaseVisionAndLanguageDataset
from .utils import demojizer_selector, twitter_preprocessor


class common_loader(BaseVisionAndLanguageDataset):
    def __init__(
            self,
            data_list: Union[List[str], str],
            root_dir: Union[List[str], str],
            splits: Union[List[str], str],
            text_processor: BertTokenizer,
            encode_kwargs: Dict[str, Any],
            weak_image_kwargs: Dict[str, Any],
            strong_image_kwargs: Dict[str, Any],
            twitter_preprocessor: Optional[Callable] = None,
            demojizer: Optional[Callable] = None,
            seed: int = 0,
            n_sup: int = 14170,
    ):
        self.twitter_preprocessor = twitter_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)

        super().__init__(
            data_list=data_list,
            root_dir=root_dir,
            splits=splits,
            text_processor=text_processor,
            encode_kwargs=encode_kwargs,
            weak_image_kwargs=weak_image_kwargs,
            strong_image_kwargs=strong_image_kwargs,
            seed=seed,
            n_sup=n_sup,
        )

    def text_preprocessor(self, texts: Union[List[str], str]) -> str:
        for ind, text in enumerate(texts):
            # 预处理文本
            remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~。]+'
            text = re.sub(remove_chars, ' ', text)
            # 去掉连续的空白符
            text = re.sub('\\s+', ' ', text)
            text = text.strip()

            texts[ind] = self.twitter_preprocessor(self.demojizer(text))
        return texts

    def text_augment(self, texts: Union[List[str], str], type: str = "weak"):
        text_aug = '[CLS] '
        for idx in range(len(texts)):
            text = texts[idx]
            if type == "strong":
                temp = text.split()
                temp2 = []
                for ttt in range(len(temp)):
                    rd = random.random()
                    if rd > 0.1:
                        temp2.append(temp[ttt])
                text = ' '.join(temp2)
            text_aug += text + ' [SEP] '
        return text_aug

    def get_image(self, eff_index: int):
        total_image = 10

        img_fn = self.image_fns[eff_index]
        images, images_aug, image_mask = [], [], []

        if self.root_dir.find('Twitter') != -1:
            cur = 1
            img_fn = img_fn[0]
            while cur <= total_image:
                file = img_fn.split('/')[1].split('.')[0] + '-' + str(cur) + '.jpg'
                image_path = os.path.join(self.root_dir, "data", img_fn.split('/')[0], img_fn.split('/')[0] + file)
                # print(img_fn, os.path.exists(image_path), image_path)
                if os.path.exists(image_path) and cur <= 10:
                    img = Image.open(image_path).convert("RGB")
                    images.append(self.weak_image_transformation(img))
                    images_aug.append(self.strong_image_transformation(img))
                    image_mask.append(1)
                cur += 1

        else:
            for image in img_fn:
                if self.root_dir.find('RPCD') != -1:
                    # RPCD needs to be processed separately
                    year = image.split('-')[0]

                    image_path = os.path.join(self.root_dir, year, 'photocritique/images', image)
                elif self.root_dir.find('TumEmo') != -1:
                    image_path = os.path.join(self.root_dir, "all_data", image + ".jpg")
                elif self.root_dir.find('Yelp') != -1:
                    image_path = os.path.join(self.root_dir, "photos", image[:2], image + ".jpg")
                else:
                    image_path = os.path.join(self.root_dir, image)

                try:
                    img = Image.open(image_path).convert("RGB")
                    images.append(self.weak_image_transformation(img))
                    images_aug.append(self.strong_image_transformation(img))
                    image_mask.append(1)
                except Exception as e:
                    continue
                if not len(images) < total_image:
                    break
        left = total_image - len(images)
        for i in range(left):
            images.append(torch.zeros((3, 224, 224), requires_grad=False))
            images_aug.append(torch.zeros((3, 224, 224), requires_grad=False))
            image_mask.append(0)

        image_mask = torch.tensor(image_mask, requires_grad=False)
        images = torch.stack(images)
        images_aug = torch.stack(images_aug)

        return images, images_aug, image_mask

    def load_dataset(self, label_type="vilt"):
        ids, labels, texts, image_fns = [], [], [], []
        with open(self.data_list, "r") as rf:
            lines = json.load(rf)
        for line in lines:
            labels.append(int(line["label"]))
            origin_id = '-'.join(line["image"])
            ids.append(origin_id)
            texts.append(line["text"])
            image_fns.append(line["image"])

        labels = torch.tensor(labels, dtype=int)

        return ids, texts, image_fns, labels


def build_common_data_loader(split, _type, all_cfg, tokenizer, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))
    cfg.update(cfg.get(_type, {}))
    # print(cfg)

    workers = cfg.get("workers", 0)
    batch_size = cfg.get("batch_size", 1)
    n_sample = cfg.get("n_sample", 1585)
    encode_kwargs = cfg.get("text_encode", None)
    weak_image_kwargs = cfg.get("transforms_weak", None)
    strong_image_kwargs = cfg.get("transforms_strong", None)

    # build dataset
    preprocessor = None
    if _type == 'Twitter':
        preprocessor = twitter_preprocessor()
    dset = common_loader(
        data_list=cfg["data_list"],
        root_dir=cfg["data_root"],
        splits=split,
        text_processor=tokenizer,
        encode_kwargs=encode_kwargs,
        weak_image_kwargs=weak_image_kwargs,
        strong_image_kwargs=strong_image_kwargs,
        twitter_preprocessor=preprocessor,
        demojizer=demojizer_selector(cfg["bert_model"]),
        seed=seed,
        n_sup=n_sample,
    )

    # build sampler
    sampler = RandomSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sampler,
        pin_memory=False,
        drop_last=split == 'train'
    )
    return loader
