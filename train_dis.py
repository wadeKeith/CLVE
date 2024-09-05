from rgbd_extractor import CLVEModel,clip_loss,clip_metrics
import pathlib
import torch
import numpy as np
import random
from lightning.fabric import Fabric
from torch import optim
from torch.utils.data import DataLoader
from CLVE_dataset import CLVEImageData, get_dataset_distributed
from tqdm import tqdm
import os, wandb, logging, yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_ema import ExponentialMovingAverage
from torch.amp import GradScaler

global_address = '/home/zxr/Documents/Github/CLVE'
DDP_FIND_UNUSED_PARAM = True
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345' 



def torch_save_atomic(what, path):
    path_tmp = path + '.tmp'
    torch.save(what, path_tmp)
    os.rename(path_tmp, path)


def setup(rank, world_size, port, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    # file_lock = f'file://home/zxr/Documents/Github/Pix2NeRF/{log_dir}/process_group_sync.lock'
    dist.init_process_group('nccl', rank=rank, world_size=world_size) # gloo init_method=None
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()
    wandb.finish()

def compute_loss(image_embed_a, image_embed_b, accumulation_steps, loss_function, temperature):
        similarity = image_embed_a @ image_embed_b.T
        loss = loss_function(similarity / temperature)
        img_a_acc, img_b_acc = clip_metrics(similarity)
        return loss / accumulation_steps, img_a_acc, img_b_acc

def train(rank, world_size, cfg):
    torch.cuda.empty_cache()

    setup(rank, world_size, 12345, 'output_dis')
    # torch.manual_seed(0)
    device = torch.device(rank)
    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    loss_function = clip_loss

    scaler = GradScaler('cuda')

    model = CLVEModel(cfg.rgbd_encoder_cfg).to(device)
    if cfg.use_ema==True:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    model_ddp = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=DDP_FIND_UNUSED_PARAM
    )
    model = model_ddp.module
    optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=cfg.learning_rate)
    model.device = device

    dataset_ddp, dataset, normalizer = get_dataset_distributed(world_size,rank,cfg.batch_size)


    total_progress_bar = tqdm(total = cfg.epochs, desc = "Total num of epochs", dynamic_ncols=True)
    total_progress_bar.update(model.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    interior_step_bar.reset(total=(len(dataset)))
    interior_step_bar.set_description(f"Progress of one epoch")
    interior_step_bar.update((model.step))

    if rank ==0:
        os.environ['WANDB_START_METHOD'] = 'thread'
        wandb_run = wandb.init(
            dir=str(os.path.join(global_address,'wandb_log')),
            # group = 'CLVE',
            mode =  'offline',
            name = 'clve',
            project = 'CLVE',
            resume = True,
            id = wandb.util.generate_id(),
        )
    for _ in range(cfg.epochs):
        total_progress_bar.update(1)
        
        step_log = dict()
        for i, (rgbd_a, rgbd_b) in enumerate(dataset_ddp):
            optimizer.zero_grad()
            if scaler.get_scale() < 1:
                scaler.update(1.)
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    norm_rgbd_a, norm_rgbd_b = normalizer.normalize(rgbd_a).to(device), normalizer.normalize(rgbd_b).to(device)
                image_embed_a, image_embed_b = model_ddp(norm_rgbd_a, norm_rgbd_b)
                loss, img_a_acc, img_b_acc  = compute_loss(image_embed_a, image_embed_b, cfg.accumulation_steps, loss_function, cfg.temperature)
            # optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if cfg.use_ema == True:
                ema.update(model_ddp.parameters())
            
            if rank==0:
                interior_step_bar.update(1)
                step_log = {
                    '-LogSoftmax': loss.item(),
                    'Image_a_Accuracy': img_a_acc.item(),
                    'Image_b_Accuracy':img_b_acc.item(),
                    'global_step': model.step,
                    'epoch': model.epoch,
                    'lr': optimizer.param_groups[0]['lr']
                }
                wandb_run.log(step_log, step=model.step)
                if model.step % cfg.save_interval == 0:
                    if cfg.use_ema == True:
                        model_dict = {
                                'ema_CLVE.pth': ema.state_dict(),
                                'CLVE.pth': model_ddp.module,
                                'optimizer.pth': optimizer.state_dict(),
                                'normalizer.pth': normalizer.state_dict(),
                            }
                    else:
                        model_dict = {
                                'CLVE.pth': model_ddp.module,
                                'optimizer.pth': optimizer.state_dict(),
                                'normalizer.pth': normalizer.state_dict(),
                            }
                    torch_save_atomic(model_dict, os.path.join(global_address,"models", f"clve_last_ckpt.pt"))
            model.step += 1
        model.epoch += 1



def main():
    config_parser = ArgumentParser()
    config_parser.add_argument("--config", default=os.path.join(global_address,'config/clve.yaml'), help="Path to config file")
    config_args = config_parser.parse_args()
    
    with open(config_args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        for key,value in config.items():
            config_parser.add_argument(f"--{key}", type=type(value), default=value)
        
    args = config_parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert num_gpus > 0, 'No GPUs found'

    mp.spawn(train, args=(num_gpus, args), nprocs=num_gpus, join=True)

    cleanup()
    

if __name__ == "__main__":
    main()

