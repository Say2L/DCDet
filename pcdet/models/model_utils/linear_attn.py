import torch
import torch.nn as nn
from einops import rearrange
from ...utils.spconv_utils import replace_feature, spconv

class FocusedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, spatial_shape, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=3):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.spatial_shape = spatial_shape
        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        #self.dwc = spconv.SubMConv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, bias=False, indice_key='subm')
    
        self.scale = nn.Parameter(torch.zeros(size=(1, dim)))

    def forward(self, x, pos, coords, voxel_inds, mask=None):
        """
        Args:
            x: input features with shape of (num_set, N, C)
            mask: (0/-inf) mask with shape of (num_set, N) or None
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        q = q + pos
        k = k + pos
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm * mask[:, :, None]

        """q, k, v = (rearrange(x, "n (h c) -> h n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("h i c, h c -> h i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("h j c, h j d -> h c d", k, v)
            x = torch.einsum("h i c, h c d, h i -> h i d", q, kv, z)
        else:
            qk = torch.einsum("h i c, h j c -> h i j", q, k)
            x = torch.einsum("h i j, h j d, h i -> h i d", qk, v, z)"""

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        
        pillar_features = rearrange(v, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        
        flatten_inds = voxel_inds.reshape(-1)

        perm = torch.arange(flatten_inds.size(0), dtype=flatten_inds.dtype, device=flatten_inds.device)
        perm = flatten_inds.new_empty(coords.size(0)).scatter_(0, flatten_inds, perm)

        pillar_features = pillar_features.reshape(-1, self.dim)[perm]

        """batch_size = torch.max(coords[:, 0]) + 1
        sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=coords[:, [0, 2, 3]].int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        sp_tensor = self.dwc(sp_tensor)"""

        x = x.reshape(-1, self.dim)[perm]
        #x = x + sp_tensor.features

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def eval(self):
        super().eval()
        print('eval')