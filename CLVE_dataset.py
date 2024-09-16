import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from itertools import combinations
from utils.normalizer import LinearNormalizer

class CLVEImageData(Dataset):
  

    def __init__(self, global_path = '/home/zxr/Documents/Github/CLVE/processed_data'):
        self.global_path = global_path
        class0 = np.load(os.path.join(self.global_path,'class0.npy'),allow_pickle=True)
        class1 = np.load(os.path.join(self.global_path,'class1.npy'),allow_pickle=True)
        class2 = np.load(os.path.join(self.global_path,'class2.npy'),allow_pickle=True)
        class3 = np.load(os.path.join(self.global_path,'class3.npy'),allow_pickle=True)
        self.idx = self.gen_index([class0.shape[0], class1.shape[0], class2.shape[0], class3.shape[0]])
        # self.samples = np.concatenate([class0.transpose([0,3,1,2]), class1.transpose([0,3,1,2]), class2.transpose([0,3,1,2]), class3.transpose([0,3,1,2])], axis=0).astype(np.float32)
        np.random.shuffle(self.idx)
        self.length = len(self.idx)

    def __len__(self):
        return self.length

    def __getitem__(self,i):
        rgbd_idx_a, rgbd_idx_b = self.idx[i][0], self.idx[i][1]
        # img_a, img_b = np.array(Image.open(os.path.join(self.global_path,'img','%d.png' %rgbd_idx_a))), np.array(Image.open(os.path.join(self.global_path,'img','%d.png' %rgbd_idx_b)))
        img_a, img_b = cv2.imread(os.path.join(self.global_path,'img','%d.png' %rgbd_idx_a)), cv2.imread(os.path.join(self.global_path,'img','%d.png' %rgbd_idx_b))
        # depth_a, depth_b = np.array(Image.open(os.path.join(self.global_path,'depth','%d.png' %rgbd_idx_a))), np.array(Image.open(os.path.join(self.global_path,'depth','%d.png' %rgbd_idx_b)))
        depth_a, depth_b = cv2.imread(os.path.join(self.global_path,'depth','%d.png' %rgbd_idx_a), cv2.IMREAD_GRAYSCALE), cv2.imread(os.path.join(self.global_path,'depth','%d.png' %rgbd_idx_b), cv2.IMREAD_GRAYSCALE)
        depth_a, depth_b = depth_a.reshape(depth_a.shape[0], depth_a.shape[1],1), depth_b.reshape(depth_b.shape[0], depth_b.shape[1],1)
        rgbd_a, rgbd_b = np.concatenate([img_a, depth_a], axis=-1).transpose([2,0,1]).astype(np.float32), np.concatenate([img_b, depth_b], axis=-1).transpose([2,0,1]).astype(np.float32)
        return rgbd_a, rgbd_b 
    
    def gen_index(self, idx_len_ls):
        start_idx = 0
        total_len = sum(idx_len_ls)
        idx_ls = []
        for idx_len in idx_len_ls:
            idx_ls.append(np.array(list(combinations(range(start_idx,start_idx+idx_len),2))))
            start_idx = start_idx+idx_len

        return np.concatenate(idx_ls,axis=0)

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        class0 = np.load(os.path.join(self.global_path,'class0.npy'),allow_pickle=True)
        class1 = np.load(os.path.join(self.global_path,'class1.npy'),allow_pickle=True)
        class2 = np.load(os.path.join(self.global_path,'class2.npy'),allow_pickle=True)
        class3 = np.load(os.path.join(self.global_path,'class3.npy'),allow_pickle=True)
        samples = np.concatenate([class0.transpose([0,3,1,2]), class1.transpose([0,3,1,2]), class2.transpose([0,3,1,2]), class3.transpose([0,3,1,2])], axis=0).astype(np.float32)
        normalizer.fit(data=samples, last_n_dims=3, mode=mode, **kwargs)
        return normalizer

def get_dataset_distributed(world_size, rank, batch_size, **kwargs):
    dataset = CLVEImageData()


    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=12,
    )

    return dataloader, dataset.get_normalizer()




if __name__ == '__main__':
    dataset = CLVEImageData()
    normalizer = dataset.get_normalizer()
    rgbd_a, rgbd_b = dataset[3]
    norm_rgbd_a, norm_rgbd_b = normalizer.normalize(rgbd_a), normalizer.normalize(rgbd_b)

    print('a')






