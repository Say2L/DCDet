import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, 
                 conv_layer=nn.Conv2d, 
                 norm_layer=nn.BatchNorm2d, 
                 act_layer=nn.ReLU, **kwargs):
        super().__init__()
        padding = kwargs.get('padding', kernel_size // 2)
        self.conv = conv_layer(inplanes, planes, kernel_size, stride, padding, bias=False)
        self.norm = norm_layer(planes, eps=0.001, momentum=0.01)
        self.act = act_layer()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3) -> None:
        super().__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()
    
    def forward(self, x):
        indentity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + indentity
        out = self.act(out)

        return out

class PillarBEVBackbone(nn.Module):
    def __init__(self, model_cfg, **kwargs) -> None:
        super().__init__()
        input_channels = model_cfg.get('BEV_CHANNELS', 256)
        self.pre_conv = BasicBlock(input_channels)
        self.conv1x1 = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, bias=False)
        self.weight = nn.Parameter(torch.randn(input_channels, input_channels, 3, 3))
        self.post_conv = ConvBlock(input_channels * 6, input_channels, kernel_size=1, stride=1)
        self.num_bev_features = input_channels

    def _forward(self, x):
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1, padding=18, dilation=18)
        x = self.post_conv(torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        return x

    def forward(self, data_dict):
        spatial_features = data_dict['multi_scale_2d_features']
        x = spatial_features['out'].dense()

        if x.requires_grad:
            x = cp.checkpoint(self._forward, x)
        else:
            x = self._forward(x)
        
        data_dict['spatial_features_2d'] = x

        return data_dict