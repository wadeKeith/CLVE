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

global_address = '/home/zxr/Documents/Github/DP_ur5e_open_door/CLVE'



class TrainCLVE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda"
        # set seed
        seed = 429
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.accumulation_steps = 1
        self.model = None
        self.optimizer = None
        self.temperature = 0.5
        self.learning_rate = 1.0e-05
        self.batch_size = 64
        self.epochs = 500
        self.loss_function = clip_loss
    
    def _compute_loss(self, image_embed_a, image_embed_b, accumulation_steps):
        similarity = image_embed_a @ image_embed_b.T
        loss = self.loss_function(similarity / self.temperature)
        img_a_acc, img_b_acc = clip_metrics(similarity)
        return loss / accumulation_steps, img_a_acc, img_b_acc

    def train(self):
        wandb_run = wandb.init(
            dir=str(os.path.join(global_address,'data')),
            # config=OmegaConf.to_container(self.cfg, resolve=True),
            group = 'CLVE',
            mode =  'offline',
            name = '429',
            project = 'CLVE',
            resume = True,
        )
        wandb.config.update(
            {
                "output_dir": os.path.join(global_address,'data'),
            }
        )
        fabric = Fabric(accelerator=self.device, devices=1, precision="32")
        fabric.launch()
        with fabric.init_module():
            self.model = CLVEModel(self.cfg)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        dataset = CLVEImageData()
        normalizer = dataset.get_normalizer()
        train_dataloader = DataLoader(dataset, self.batch_size, True)
        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

        best_train_loss = None
        self.global_step = 0

        for epoch in range(self.epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar1 = tqdm(train_dataloader)
            epoch_train_loss = 0.0
            step_log = dict()
            for i, (rgbd_a, rgbd_b) in enumerate(pbar1):
                norm_rgbd_a, norm_rgbd_b = normalizer.normalize(rgbd_a).detach().to(self.device), normalizer.normalize(rgbd_b).detach().to(self.device)
                image_embed_a, image_embed_b = self.model(norm_rgbd_a, norm_rgbd_b)
                loss, img_a_acc, img_b_acc  = self._compute_loss(image_embed_a, image_embed_b, self.accumulation_steps)
                fabric.backward(loss)
                if (i+1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar1.set_postfix({'-LogSoftmax':loss.item(), 'Image_a_Accuracy':img_a_acc.item(),'Image_b_Accuracy':img_b_acc.item()})
                step_log = {
                        '-LogSoftmax': loss.item(),
                        'Image_a_Accuracy': img_a_acc.item(),
                        'Image_b_Accuracy':img_b_acc.item(),
                        'global_step': self.global_step,
                        'epoch': epoch,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                wandb_run.log(step_log, step=self.global_step)
                epoch_train_loss += loss.item()*len(rgbd_a)
                self.global_step += 1
            
            epoch_train_loss /= len(dataset)
            if best_train_loss is None or epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                torch.save(self.model.state_dict(), os.path.join(global_address,"models", f"clve_best_train_loss_ckpt.pt"))
                torch.save(self.optimizer.state_dict(), os.path.join(global_address, "models", f"clve_best_train_loss_optim.pt"))
            torch.save(self.model.state_dict(), os.path.join(global_address,"models", f"clve_latest_ckpt.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(global_address,"models", f"clve_latest_optim.pt"))








def main():
    config_parser = ArgumentParser()
    config_parser.add_argument("--config", default=os.path.join(global_address,'config/clve.yaml'), help="Path to config file")
    config_args = config_parser.parse_args()
    
    with open(config_args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        for key,value in config.items():
            config_parser.add_argument(f"--{key}", type=type(value), default=value)
        
    args = config_parser.parse_args()

    train_CLVE = TrainCLVE(args.policy)
    
    train_CLVE.train()

if __name__ == "__main__":
    main()

