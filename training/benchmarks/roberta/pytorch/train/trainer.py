# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import os
import sys

import torch

from model import create_model
from optimizers import create_optimizer
from schedulers import create_scheduler

from driver import Driver, dist_pytorch

class Trainer:
    def __init__(self, driver, adapter, evaluator, tokenizer, training_state, device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator
        self.tokenizer = tokenizer

    def init(self):
        device = torch.device(self.config.device)
        dist_pytorch.main_proc_print("Init progress: ")
        self.model = create_model()
        self.model.to(self.device)

        self.optimizer = create_optimizer(self.model, self.config)
        self.lr_scheduler = create_scheduler(self.config, self.model)

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def train_one_epoch(self, train_dataloader):
        model = self.model
        optimizer = self.optimizer
        data_loader = train_dataloader
        device = self.device
        state = self.training_state
        config = self.config
        epoch = state.epoch

        print(f'Epoch + {str(epoch + 1)}')

        if self.config.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        model.train()
        noeval_start_time = time.time()

        # metric_logger = utils.utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter(
        #     'lr', utils.utils.SmoothedValue(window_size=1, fmt='{value:.6f}')
        # )
        # header = 'Epoch: [{}]'.format(epoch)

        for step, batch in enumerate(data_loader):
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.step()
            self.lr_scheduler.step()
            optimizer.zero_grad()

            print(f'Loss: {str(float(loss))}')

    def evaluate(self, model, data_loader, device):
        self.model.eval()
        self.evaluator.reset()
        for step, batch in enumerate(data_loader):
            print(f'.')
     
