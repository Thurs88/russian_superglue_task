from typing import List

from omegaconf import DictConfig

from src.technical_utils import load_obj
from src.text_utils import prepare_data


def get_test_dataset(cfg: DictConfig, test_data: List, tokenizer):
    """
    Get test dataset
    :param cfg:
    :param test_data:
    :param tokenizer:
    :return:
    """
    # test_data = prepare_data(test_data)
    dataset_class = load_obj(cfg.inference.dataset_class)
    test_dataset = dataset_class(
        data=test_data,
        tokenizer=tokenizer,
    )
    return test_dataset
