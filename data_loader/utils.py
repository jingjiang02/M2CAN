import os
import sys
import warnings
from typing import Dict, List, Optional, Callable, Tuple

import torch
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from emoji import demojize
from transformers import ViltProcessor, AutoTokenizer


class VaultProcessor(ViltProcessor):
    @classmethod
    def from_pretrained(
            self, vilt_directory: str, bert_directory: Optional[str] = None
    ):
        try:
            processor = super().from_pretrained(vilt_directory)
        except:
            # not all processors have been implemented
            processor = super().from_pretrained("dandelin/vilt-b32-mlm")
        if bert_directory is not None:
            processor.tokenizer = AutoTokenizer.from_pretrained(bert_directory)
        return processor


def safe_dict_concat(
        dict_list: List[Dict[str, torch.Tensor]], dim: int = 0
) -> Dict[str, torch.Tensor]:
    """Takes a list of `transformer`-style inputs and concatenates
    the tensors found in them by zero-padding the attention mask.

    Args:
        dict_list: list of `transformers` input dicts.
        dim: dimension to concatenate (should probably remain 0 (default)).

    Returns:
        Single dict with batched inputs.
    """

    w_max = max([inp["pixel_values"].size(-2) for inp in dict_list])
    h_max = max([inp["pixel_values"].size(-1) for inp in dict_list])

    for inp in dict_list:
        for key in list(inp):
            if "pixel" not in key:
                continue
            value = inp[key]
            new_value = torch.zeros(*value.shape[:-2], w_max, h_max)
            new_value[..., : value.shape[-2], : value.shape[-1]] = value
            inp[key] = new_value

    return {
        k: torch.concat([d[k] for d in dict_list], dim=dim)
        for k in dict_list[0]
    }


LOGGING_FORMAT = "%(levelname)s-%(name)s(%(asctime)s)   %(message)s"


def demojizer_selector(
        model_name: str, delimiters: Tuple[str] = ("(", ")")
) -> Callable:
    """Fetches demojizer based on model.

    Args:
        model_name: language model name (from `transformers`).
        delimiters: strings to delimit the emoji description by.

    Returns:
        Demojizer function (identity if correspondence not set).
    """
    demojizers = {
        "vinai/bertweet-base": lambda x: x,
        "bert-base-uncased": lambda x: demojize(
            x, language="en", delimiters=delimiters
        ).replace("_", " "),
        "Yanzhu/bertweetfr-base": lambda x: demojize(
            x, language="fr", delimiters=delimiters
        ).replace("_", " "),
        "flaubert/flaubert_base_uncased": lambda x: demojize(
            x, language="fr", delimiters=delimiters
        ).replace("_", " "),
        "dccuchile/bert-base-spanish-wwm-uncased": lambda x: demojize(
            x, language="es", delimiters=delimiters
        ).replace("_", " "),
        "asafaya/bert-base-arabic": lambda x: x,
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": lambda x: x,
        "bert-base-multilingual-uncased": lambda x: demojize(
            x, language="en", delimiters=delimiters
        ).replace("_", " "),
    }
    return demojizers.get(model_name, lambda x: x)


def twitter_preprocessor(
        normalized_tags: Optional[List] = None, extra_tags: Optional[List] = None
) -> Callable:
    """Creates a Twitter specific text preprocessor.

    Args:
        normalized_tags: tags to anonymize, e.g. @userNamE -> user.
        extra_tags: other normalizations, e.g. Helloooooo -> hello.

    Returns:
        A function that accepts a string and returns the
        processed string.
    """

    normalized_tags = normalized_tags or ["url", "email", "phone", "user"]

    extra_tags = extra_tags or [
        "hashtag",
        "elongated",
        "allcaps",
        "repeated",
        "emphasis",
        "censored",
    ]

    def intersect_delimiters(l: List[str], demiliter: str) -> List[str]:
        new_l = []
        for elem in l:
            new_l.extend([elem, demiliter])
        return new_l

    def tag_handler_and_joiner(tokens: List[str]) -> str:
        new_tokens = []
        for token in tokens:
            for tag in normalized_tags:
                if token == f"<{tag}>":
                    token = tag
            for tag in set(extra_tags).difference(["hashtag"]):
                if token in (f"<{tag}>", f"</{tag}>"):
                    token = None
            if token:
                new_tokens.append(token)

        full_str = []
        end_pos = -1

        if "hashtag" in extra_tags:
            start_pos = -1
            while True:
                try:
                    start_pos = new_tokens.index("<hashtag>", start_pos + 1)
                    full_str.extend(
                        intersect_delimiters(
                            new_tokens[end_pos + 1: start_pos], " "
                        )
                    )
                    end_pos = new_tokens.index("</hashtag>", start_pos + 1)
                    full_str.extend(
                        ["# "]
                        + intersect_delimiters(
                            new_tokens[start_pos + 1: end_pos], "-"
                        )[:-1]
                        + [" "]
                    )
                except:
                    break

        full_str.extend(intersect_delimiters(new_tokens[end_pos + 1:], " "))
        return "".join(full_str).strip()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # stop ekphrasis prints
        sys.stdout = open(os.devnull, "w")

        preprocessor = TextPreProcessor(
            normalize=normalized_tags,
            annotate=extra_tags,
            all_caps_tag="wrap",
            fix_text=False,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        ).pre_process_doc

        # re-enable prints
        sys.stdout = sys.__stdout__

    fun = lambda x: tag_handler_and_joiner(preprocessor(x))
    fun.log = f"ekphrasis: {normalized_tags}, {extra_tags} | tag handler"
    return fun
