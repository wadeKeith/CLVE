import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import time
from torchsummary import summary

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from DFormer import DFormer_Base, DFormer_Large, DFormer_Small, DFormer_Tiny


class QuickGELU(torch.nn.Module):
   def forward(self, x: torch.Tensor):
       return x * torch.sigmoid(1.702 * x)
   
class CLVEAttentiveLayer(nn.Module):
    def __init__(self, n_head, d_embed):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_embed)
        # self.attention = SelfAttention(n_head, d_embed)
        self.attention = nn.MultiheadAttention(d_embed, n_head, batch_first=True)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)
        self.activation = QuickGELU()

    def forward(self, x, causal_mask):
        residue = x
        x = self.layernorm_1(x)
        # x = self.attention(x, causal_mask = True)
        x, _ = self.attention(x, x, x, is_causal = True, attn_mask = causal_mask) if causal_mask is not None else self.attention(x, x, x)
        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x += residue
        return x
   
class CLVEProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        input_dim = input_dim
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = QuickGELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        x = self.layer_norm(x)
        return x 

class CLVE(nn.Module):
    def __init__(self, 
                 rgbd_encoder_cfg = None
                 ):
        super().__init__()
        self.rgbd_network_backbone = rgbd_encoder_cfg['rgbd_network_backbone']
        if self.rgbd_network_backbone == 'DFormer_Tiny':
            self.extractor = DFormer_Tiny(pretrained = True)
            self.d_embed = 384
        elif self.rgbd_network_backbone == 'DFormer_Small':
            self.extractor = DFormer_Small(pretrained = True)
            self.d_embed = 768
        elif self.rgbd_network_backbone == 'DFormer_Base':
            self.extractor = DFormer_Base(pretrained = True)
            self.d_embed = 768
        elif self.rgbd_network_backbone == 'DFormer_Large':
            self.extractor = DFormer_Large(pretrained = True)
            self.d_embed = 864
        else:
            raise NotImplementedError(f"rgbd_network_backbone: {self.rgbd_network_backbone}")
        del self.extractor.pred
        self.extractor.eval()
        # self.extractor.requires_grad_(False)
        for p in self.extractor.parameters():
            p.requires_grad = False
        
        self.image_size = 300
        self.attention_layers = nn.ModuleList([
            CLVEAttentiveLayer(rgbd_encoder_cfg['num_heads'], self.d_embed) 
            for i in range(rgbd_encoder_cfg['num_layers_clve_attentive'])])
        self.linear = nn.Linear(self.image_size , 1)
        self.projection_head = CLVEProjectionHead(input_dim=self.d_embed, output_dim=rgbd_encoder_cfg['out_channels'], dropout=rgbd_encoder_cfg['dropout'])


    def forward(self, rgbd) -> torch.Tensor:
        with torch.no_grad():
            rgbd_feat = self.extractor(rgbd)
        batch, channels, h, w = rgbd_feat.shape[0], rgbd_feat.shape[1], rgbd_feat.shape[2], rgbd_feat.shape[3]
        rgbd_feat = rgbd_feat.view(batch, channels, h *w).swapaxes(1,2)
        for layer in self.attention_layers:
            state = layer(rgbd_feat, None)   # causal mask not needed for image features
        out = self.linear(state.swapaxes(2,1)).squeeze(-1)
        out = self.projection_head(out)
        # out = out.swapaxes(1,2).mean([-1])
        return out


class CLVEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_encoder = CLVE(cfg['rgbd_encoder_cfg'])

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, image_a, image_b):
        image_embed_a = self.image_encoder(image_a)
        image_embed_b = self.image_encoder(image_b)
        return image_embed_a, image_embed_b
    
    
def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity):
    text_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (text_loss + image_loss) / 2.0

def clip_metrics(similarity):
    y = torch.arange(len(similarity)).to(similarity.device)
    imgb2imga_match_idx = similarity.argmax(dim=1)
    imga2imgb_match_idx = similarity.argmax(dim=0)

    img_b_acc = (imgb2imga_match_idx == y).float().mean()
    img_a_acc = (imga2imgb_match_idx == y).float().mean()

    return img_a_acc, img_b_acc

if __name__ == '__main__':
    clve = CLVE()
    x = torch.randn([1,4,480,640])
    y = clve(x)

    print('a')


