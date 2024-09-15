from rgbd_extractor import CLVEModel,clip_loss,clip_metrics
import pathlib
import torch
import numpy as np
import random
from lightning.fabric import Fabric
from torch import optim
from torch.utils.data import DataLoader
from CLVE_dataset import CLVEImageData, get_dataset_distributed
import os

global_address = '/home/zxr/Documents/Github/CLVE/models'


checkpoint = torch.load(os.path.join(global_address, 'clve_last_epoch_ckpt.pth'))


print('a')

