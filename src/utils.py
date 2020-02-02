import shutil
import random
import logging
from pprint import pformat
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import torch
import numpy as np
import pandas as pd


def now():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    
def load_csv(path):
    return pd.read_csv(path)


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


def get_lr(optimizer):
    lr = list()
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    if len(lr) == 1:
        return lr[0]
    else:
        return lr


class Logger:
    """Logging Uitlity Class for monitoring and debugging
    """

    def __init__(self,
                 name,
                 log_fname,
                 log_level=logging.INFO,
                 custom_log_handler=None):

        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        ch = logging.FileHandler(log_fname)
        self.logger.addHandler(ch)
        self.logger.addHandler(logging.StreamHandler())

        if custom_log_handler:
            if isinstance(custom_log_handler, list):
                for handler in custom_log_handler:
                    self.logger.addHandler(handler)
            else:
                self.logger.addHandler(handler)

    def kiritori(self):
        self.logger.info('-'*80)

    def double_kiritori(self):
        self.logger.info('='*80)

    def space(self):
        self.logger.info('\n')

    @contextmanager
    def interval_timer(self, name):
        start_time = datetime.now()
        self.logger.info("\n")
        self.logger.info(f"Execution {name} start at {start_time}")
        try:
            yield
        finally:
            end_time = datetime.now()
            td = end_time - start_time
            self.logger.info(f"Execution {name} end at {end_time}")
            self.logger.info(f"Execution Time : {td}")
            self.logger.info("\n")

    def __getattr__(self, attr):
        """
        for calling logging class attribute
        if you call attributes of other class, raise AttributeError
        """
        # self.logger.info(f"{datetime.now()}")
        return getattr(self.logger, attr)


# setup for kernel
def setup(exp_name, config):
    """init experiment (directory setup etc...)"""

    result_dir = Path(f'result/{exp_name}/')
    result_dir.mkdir(parents=True)
    shutil.copytree("src", result_dir / "src")

    set_seed(config.seed)

    device = torch.device(config.device_name)

    log = Logger(exp_name, result_dir / 'exp.log')

    log.info("configuration is following...")
    log.info(pformat(config.__dict__))

    return device, log, result_dir
