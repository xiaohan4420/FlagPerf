# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

from dataloaders.dataloader import prepare_raw_dataset, \
preprocessing_datasets, prepare_train_dataset, prepare_eval_dataset

from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from model import create_tokenizer

logger = None

def main() -> Tuple[Any, Any]:
    global logger
    global config

    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    config.distributed = dist_pytorch.get_world_size() > 1
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    
    # init起始时间，单位ms
    init_start_time = logger.previous_log_time  

    # 构建数据集
    raw_datasets = prepare_raw_dataset(config)

    tokenizer = create_tokenizer(config)

    tokenized_datasets = preprocessing_datasets(config, tokenizer, raw_datasets)
    train_dataset = prepare_train_dataset(config, tokenized_datasets)
    eval_dataset = prepare_eval_dataset(config, tokenized_datasets)

    # 训练前确定seed
    seed = config.seed

    init_helper.set_seed(seed, model_driver.config.vendor)

    # 创建TrainingState对象
    training_state = TrainingState()

    # init evaluator
    evaluator = Evaluator(config, "accuracy")
    evaluator.reset()

    trainer = Trainer(
        driver=model_driver,
        adapter=trainer_adapter,
        evaluator=evaluator,
        tokenizer=tokenizer,
        training_state=training_state,
        device=config.device,
        config=config,
    )
    training_state._trainer = trainer
     
    if config.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if config.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({config.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(config.max_seq_length, tokenizer.model_max_length)   

    padding = "max_length" if config.pad_to_max_length else False

    # 设置分布式环境

    # init eval:
    init_evaluation_start = time.time()

    trainer.evaluate(trainer.model, eval_dataset, device=trainer.device)

    init_evaluation_end = time.time()

    if not config.do_train:
        return config, training_state
    
    # TRAIN_START

    # 训练过程
    epoch = 0
    while not training_state.end_training:
        training_state.epoch = epoch
        trainer.train_one_epoch()
        epoch += 1
    
    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)
    
    e2e_time = time.time() - start
    if config_update.do_train:
        
        finished_info = {
            "e2e_time": e2e_time,
        }
    else:
        finished_info = {
            "e2e_time": e2e_time,
        }
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)