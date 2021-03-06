#!/usr/env/bin python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import sys
import subprocess
import os
import yaml

import chainer
from chainer import cuda, optimizers, serializers
from chainer import training
from chainercv.links import PixelwiseSoftmaxClassifier

from linknet.config_utils import *

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


from collections import OrderedDict
yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

from linknet.models import linknet_paper

def train_linknet():
    """Training LinkNet."""
    chainer.config.debug = True
    config = parse_args()
    train_data, test_data = load_dataset(config["dataset"])
    train_iter, test_iter = create_iterator(train_data, test_data, config['iterator'])
    model = get_model(config["model"])
    class_weight = get_class_weight(config)
    model = PixelwiseSoftmaxClassifier(model, class_weight=class_weight)
    optimizer = create_optimizer(config['optimizer'], model)
    devices = parse_devices(config['gpus'])
    updater = create_updater(train_iter, optimizer, config['updater'], devices)
    trainer = training.Trainer(updater, config['end_trigger'], out=config['results'])
    trainer = create_extension(trainer, test_iter,  model.predictor,
                               config['extension'], devices=devices)
    trainer.run()
    chainer.serializers.save_npz(os.path.join(config['results'], 'model.npz'),
                                 model.predictor)

def main():
    train_linknet()

if __name__ == '__main__':
    main()
