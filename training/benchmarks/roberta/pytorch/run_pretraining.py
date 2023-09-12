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

from dataloaders.dataloader import prepare_train_dataset, build_train_dataloader \
, prepare_raw_dataset

from train import trainer_adapter
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState

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
    init_start_time = logger.previous_log_time  # init起始时间，单位ms

    # 构建数据集
    train_dataset = prepare_train_dataset(config)
    tokenizer = create_tokenizer
    train_dataset = RobertaDataset(train_dataset, tokenizer)

    raw_dataset = 

    train_dataloader = build_train_dataloader(train_dataset, config)

    # train_dataset = build_train_dataset(config)
    # eval_dataset = build_eval_dataset(config)
    # train_dataloader = build_train_dataloader(train_dataset, config)
    # eval_dataloader = build_eval_dataloader(eval_dataset, config)

    seed = config.seed

    init_helper.set_seed(seed, model_driver.config.vendor)

    # 创建TrainingState对象
    training_state = TrainingState()

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
     
    # processing the datasets.
    if config.do_train:
        column_names = list(raw_dataset)

    # 设置分布式环境


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