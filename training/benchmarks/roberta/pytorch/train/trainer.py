# Copyright (c) 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import torch
import sys

from model import create_model

from driver import Driver, dist_pytorch

class Trainer:
    def __init__(self, driver, adapter, evaluator, training_state, device, config):
        super(Trainer, self).__init__()
        self.driver = driver
        self.adapter = adapter
        self.training_state = training_state
        self.device = device
        self.config = config
        self.evaluator = evaluator

    def init(self):
        device = torch.device(self.config.device)
        dist_pytorch.main_proc_print("Init progress: ")
        self.model = create_model()
        self.model.to(self.device)

    def train_one_epoch(self, train_dataloader):
        model = self.model
        optimizer = self.optimizer
        data_loader = train_dataloader
        device = self.device
        state = self.training_state
        config = self.config
        epoch = state.epoch

        if self.config.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        model.train()
        noeval_start_time = time.time()
        metric_logger = utils.utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            'lr', utils.utils.SmoothedValue(window_size=1, fmt='{value:.6f}')
        )
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None
        if epoch == 0:
            warmup_factor
     
