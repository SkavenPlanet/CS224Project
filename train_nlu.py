from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager
from nemo.utils import logging

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

DATA_DIR = "datasets/nlu"
NEMO_DIR = '.'
BRANCH = 'r1.8.2'
#wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/intent_slot_classification/conf/intent_slot_classification_config.yaml', NEMO_DIR)

# print content of the config file
config_file = "intent_slot_classification_config.yaml"
config = OmegaConf.load(config_file)

config.model.data_dir = f'{DATA_DIR}'

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1
config.trainer.accelerator = accelerator

config.trainer.precision = 16 if torch.cuda.is_available() else 32

# for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
# config.trainer.amp_level = O1

# remove distributed training flags
config.trainer.strategy = None

# setup a small number of epochs for demonstration purposes of this tutorial
config.trainer.max_epochs = 5

trainer = pl.Trainer(**config.trainer)

#config.exp_manager.exp_dir = 'nemo_experiments/IntentSlot/2022-05-21_18-19-07'
#print(OmegaConf.to_yaml(config))

#exp_dir = exp_manager(trainer, config.get("exp_manager", None))
# the exp_dir provides a path to the current experiment for easy access
#print(str(exp_dir))

# initialize the model
model = nemo_nlp.models.IntentSlotClassificationModel(config.model, trainer=trainer)
# train
trainer.fit(model)