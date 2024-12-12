import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


class BaseVisionAndLanguageDataset(Dataset, ABC):
    """Pytorch base dataset for Vision and Language tasks.
    Attributes:
        data_list: list file of dataset.
        root_dir: root directory of dataset.
        splits: which split(s) to use.
        text_processor: text "tokenizer".
        ids: identifiers of examples.
        texts: list of sentences per example.
        image_fns: corresponding image filenames.
        labels: corresponding labels.
        effective_inds: because we may have multiple texts per image
            or vice versa, we have tuples in our data structure. To
            allow the user to use simple integers, we map them to
            tuple indices.
        image_preprocessor: image preprocessing.
        encode_kwargs: kwargs for the tokenization process.
    """

    def __init__(
            self,
            data_list: Union[List[str], str],
            root_dir: Union[List[str], str],
            splits: Union[List[str], str],
            text_processor: BertTokenizer,
            encode_kwargs: Dict[str, Any],
            weak_image_kwargs: Dict[str, Any],
            strong_image_kwargs: Dict[str, Any],
            seed: int = 0,
            n_sup: int = 14170,
    ):
        self.logger = logging.getLogger("global")
        self.data_list = data_list
        self.root_dir = root_dir
        self.splits = splits
        self.text_processor = text_processor
        self.encode_kwargs = encode_kwargs

        (
            self.ids,
            self.texts,
            self.image_fns,
            self.labels,
            *kwargs,
        ) = self.load_dataset()

        self.get_effective_inds(seed, n_sup)

        self.weak_image_transformation = self._init_image_transformation(weak_image_kwargs)
        self.strong_image_transformation = self._init_image_transformation(strong_image_kwargs)

        self.num_sample = len(self.effective_inds)
        assert self.num_sample > 0
        self.logger.info("# samples: {}".format(self.num_sample))

    def get_effective_inds(self, seed, n_sup):
        self.effective_inds = [i for i, example_texts in enumerate(self.texts)]
        random.seed(seed)
        if len(self.effective_inds) < n_sup:
            print('-' * 50 + f'Insufficient sample! total:{len(self.effective_inds)} < expected:{n_sup}' + '-' * 50)
            num_repeat = math.ceil(n_sup / len(self.effective_inds))
            self.effective_inds = self.effective_inds * num_repeat
        elif len(self.effective_inds) > n_sup:
            print('-' * 50 + f'There are too many samples, sampling is required! total:{len(self.effective_inds)} > expected:{n_sup}' + '-' * 50)
        self.effective_inds = random.sample(self.effective_inds, n_sup)

    def __len__(self):
        return len(self.effective_inds)

    @abstractmethod
    def load_dataset(
            self
    ) -> Tuple[
        List[str],
        Union[List[str], List[List[str]]],
        Union[List[str], List[List[str]]],
        torch.Tensor,
    ]:
        """Loads dataset. Returns IDs, texts, image filenames and labels."""

    @abstractmethod
    def text_preprocessor(
            self,
            texts: Union[List[str], str],
    ) -> Union[str, List[str]]:
        """text preprocess."""

    @abstractmethod
    def text_augment(
            self,
            texts: Union[List[str], str],
            type: str = "weak",
    ) -> Union[str, List[str]]:
        """text augment."""

    def get_text(self, eff_index: int):
        texts = self.text_preprocessor(self.texts[eff_index])
        return self.text_augment(texts, "weak"), self.text_augment(texts, "strong")

    @abstractmethod
    def get_image(self, eff_index: int):
        """ get images ,whitch can be one image(B, 3, H, W) or a group of images(B, l, 3, H, W)"""

    def get_label(self, eff_index: int) -> torch.Tensor:
        return self.labels[eff_index]

    def text_encode(
            self, text, text_aug,
    ):
        max_length = self.encode_kwargs.get('max_length', 200)
        inputs = {}
        text_dict = self.text_processor.encode_plus(text, max_length=max_length, pad_to_max_length=True,
                                                    add_special_tokens=True, return_attention_mask=True,
                                                    truncation=True)
        text_dict_aug = self.text_processor.encode_plus(text_aug, max_length=max_length, pad_to_max_length=True,
                                                        add_special_tokens=True, return_attention_mask=True,
                                                        truncation=True)
        inputs.update({"input_ids": torch.tensor(text_dict['input_ids'], requires_grad=False)})
        inputs.update({"token_type_ids": torch.tensor(text_dict['token_type_ids'], requires_grad=False)})
        inputs.update({"attention_mask": torch.tensor(text_dict['attention_mask'], requires_grad=False)})
        inputs.update({"input_ids_aug": torch.tensor(text_dict_aug['input_ids'], requires_grad=False)})
        inputs.update({"token_type_ids_aug": torch.tensor(text_dict_aug['token_type_ids'], requires_grad=False)})
        inputs.update({"attention_mask_aug": torch.tensor(text_dict_aug['attention_mask'], requires_grad=False)})
        return inputs

    def __getitem__(self, index: int):
        eff_index = self.effective_inds[index]
        id = self.ids[eff_index]
        text, text_aug = self.get_text(eff_index)
        inputs = self.text_encode(text, text_aug)
        image, image_aug, image_mask = self.get_image(eff_index)
        inputs.update({
            "image": image,
            "image_aug": image_aug,
            "image_mask": image_mask,
        })
        inputs.update({"label": self.get_label(eff_index)})
        inputs.update({"id": id})
        return inputs

    def _init_image_transformation(self, cfg):
        """Returns image transformation used for augmentation, etc."""
        trs_form = []
        mean, std = cfg["mean"], cfg["std"]
        if cfg.get("resize", False):
            trs_form.append(transforms.Resize(cfg["resize"]))

        if cfg.get("rand_resize_crop", False):
            trs_form.append(transforms.RandomResizedCrop(cfg["rand_resize_crop"]))

        if cfg.get("center_crop", False):
            trs_form.append(transforms.CenterCrop(cfg["center_crop"]))

        if cfg.get("rand_flip", False):
            trs_form.append(transforms.RandomHorizontalFlip(p=cfg["rand_flip"]))

        if cfg.get("rand_apply", False):
            rand_apply_transforms = []
            if cfg["rand_apply"].get("color_jitter", False):
                cfg_color = cfg["rand_apply"]["color_jitter"]
                rand_apply_transforms.append(
                    transforms.ColorJitter(
                        brightness=cfg_color["brightness"],
                        contrast=cfg_color["contrast"],
                        saturation=cfg_color["saturation"],
                        hue=cfg_color["hue"],
                    )
                )

            trs_form.append(transforms.RandomApply(rand_apply_transforms, p=cfg["rand_apply"]["p"]))

        if cfg.get("rand_gray_scale", False):
            trs_form.append(transforms.RandomGrayscale(p=cfg["rand_gray_scale"]))

        trs_form.append(transforms.ToTensor())
        trs_form.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(trs_form)
