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
from torch_ema import ExponentialMovingAverage
from utils.normalizer import LinearNormalizer
import cv2
import os, wandb, logging, yaml
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP

global_address = '/home/zxr/Documents/Github/CLVE'
DDP_FIND_UNUSED_PARAM = True
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '2950' 
# similarity_dict = {'0':[],
#                  '1':[],
#                  '2':[],
#                  '3':[]}



def setup(rank, world_size, port, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    # file_lock = f'file://home/zxr/Documents/Github/Pix2NeRF/{log_dir}/process_group_sync.lock'
    dist.init_process_group('nccl', rank=rank, world_size=world_size) # gloo init_method=None
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    # wandb.finish()
def eval(rank, world_size, cfg):
    torch.cuda.empty_cache()

    setup(rank, world_size, 12345, 'output_dis')
    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # set device
    device = torch.device(rank)
    # set global var
    similarity_ls = []

    checkpoint = torch.load(os.path.join(global_address, 'models/clve_last_step_ckpt.pt'), map_location=device, weights_only=False)

    model = checkpoint['CLVE.pth'].to(device)
    model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema_decay)
    model_ema.load_state_dict(checkpoint['ema_CLVE.pth'])
    model_ema = model_ema
    model_ema.copy_to(model.parameters())

    model.eval()
    model.to(device)

    model_ddp = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=DDP_FIND_UNUSED_PARAM
    )
    model_ddp.eval()
    dataset = CLVEImageData()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size,
        shuffle=False,
        # drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    # dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
    #                          shuffle=False,
    #                            num_workers=cfg.num_workers,
    #                            pin_memory=True)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(checkpoint['normalizer.pth'])
    # normalizer = dataset.get_normalizer()
    if rank ==0:
        pbar1 = tqdm(total=len(dataloader), desc="Progress of one dataset" , dynamic_ncols=True)
    
    for i, (rgbd_a, rgbd_b) in enumerate(dataloader):
        norm_rgbd_a, norm_rgbd_b = normalizer(rgbd_a), normalizer(rgbd_b)

        embed_a, embed_b = model_ddp(norm_rgbd_a.to(device), norm_rgbd_b.to(device))
        similarity =torch.diag(embed_a @ embed_b.T).detach().cpu().numpy()
        similarity_ls.append(similarity)
        if rank ==0:
            pbar1.update(1)
        # if i ==3:
        #     break

    similarity_array = np.concatenate(similarity_ls, axis=0)
    # assert len(similarity_array) == len(dataset)
    np.save('similarity_%d.npy' %rank, similarity_array)

def main():
    config_parser = ArgumentParser()
    config_parser.add_argument("--config", default=os.path.join(global_address,'config/clve_eval.yaml'), help="Path to config file")
    config_args = config_parser.parse_args()
    
    with open(config_args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        for key,value in config.items():
            config_parser.add_argument(f"--{key}", type=type(value), default=value)
        
    args = config_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert num_gpus > 0, 'No GPUs found'

    # global similarity_ls

    mp.spawn(eval, args=(num_gpus, args), nprocs=num_gpus, join=True)

    


    # cleanup()

    

if __name__ == "__main__":
    main()
