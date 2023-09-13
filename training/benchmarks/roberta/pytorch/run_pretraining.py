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

    raw_dataset = prepare_raw_dataset(config)

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
        column_names = list(raw_dataset["train"].features)
    else:
        column_names = list(raw_dataset["validation"].features)
    text_column_name = "text" if "text"in column_names else column_names[0]

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

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        tokenizer = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        return tokenizer

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=[text_column_name],
        desc="Running tokenizer on dataset line_by_line."
    )

    if config.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = tokenized_datasets["train"]
        if config.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), config.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if config.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset.")
        eval_dataset = tokenized_datasets["validation"]
        if config.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), config.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logtis, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
    
    metric = evaluate.load("accuracy")
    



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