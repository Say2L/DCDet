import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from einops import rearrange 
import spconv.pytorch as spconv
from torch.utils.checkpoint import checkpoint

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats)
            )

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
    
class UpFormerBlock_linear_attn(nn.Module):
    def __init__(self, in_channel, out_channel, stride=[6, 6, 4], heads=8, focusing_factor=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = np.array(stride)
        self.shift = self.stride // 2
        self.heads = heads

        self.pos_embedding = PositionEmbeddingLearned(3, out_channel)

        self.pre_embedding = nn.Sequential(
            nn.Linear(self.in_channel, self.out_channel),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01))
        
        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(self.out_channel, self.out_channel * 3)
        
        self.proj = nn.Linear(self.out_channel, self.out_channel)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = spconv.SubMConv3d(self.out_channel, self.out_channel, 3)
        
        self.scale = nn.Parameter(torch.zeros(size=(1, self.out_channel)))

        self.linear1 = nn.Linear(self.out_channel, self.out_channel)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.out_channel, self.out_channel)

        self.norm1 = nn.LayerNorm(self.out_channel, eps=1e-3)
        self.norm2 = nn.LayerNorm(self.out_channel, eps=1e-3)



    def forward(self, sp_tensors):
        """
            ms_sp_tensors: multi scale SparseConv tensors
            kernel_size: output feature map stride

            return: multi scale fusion features 
        """
        
        eps = 1e-4
        raw_sp_features = sp_tensors.features
        sp_indices = sp_tensors.indices
        #pos_weight = self.pos_weight((torch.pi / 2) * torch.sigmoid(ms_indices[:, 1:].to(ms_features.dtype)))
        sp_features = self.pre_embedding(raw_sp_features)
        pos_embed = self.pos_embedding(sp_indices[:, 1:].to(sp_features.dtype))

        N, C = sp_features.shape
        qkv = self.qkv(sp_features).reshape(N, 3, C).permute(1, 0, 2)
        q, k, v = qkv.unbind(0)
        k = k + pos_embed

        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-4
        k = kernel_function(k) + 1e-4
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(self.focusing_factor) <= 6:
            q = q ** self.focusing_factor
            k = k ** self.focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** self.focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "n (h c) -> h n c", h=self.heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("h i c, h c -> h i", q, k.sum(dim=1)) + 1e-4)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("h j c, h j d -> h c d", k, v)
            x = torch.einsum("h i c, h c d, h i -> h i d", q, kv, z)
            
        else:
            qk = torch.einsum("h i c, h j c -> h i j", q, k)
            x = torch.einsum("h i j, h j d, h i -> h i d", qk, v, z)
        
        x = rearrange(x, 'h n d -> n (h d)')
        v = rearrange(v, 'h n d -> n (h d)')
        sp_tensors = sp_tensors.replace_feature(v)

        sp_tensors = self.dwc(sp_tensors)

        x += sp_tensors.features
        x = self.proj(x)
        x = self.norm1(x + sp_features)

        """x1 = self.linear1(x)
        x1 = self.linear2(self.activation(x1))

        x += x1
        x = self.norm2(x)"""

        sp_tensors = sp_tensors.replace_feature(x)
        
        return sp_tensors