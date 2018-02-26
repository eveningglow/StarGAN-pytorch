import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='relu'):
        super(ConvBlock, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]
        
        if norm is True:
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]
            
        if non_linear == 'relu':
            layers += [nn.ReLU()]
        elif non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU()]
        elif non_linear == 'tanh':
            layers += [nn.Tanh()]
            
        self.conv_block = nn.Sequential(* layers)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out
        
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1):
        super(ResBlock, self).__init__()
        
        # Use 2 ConvBlock in 1 ResBlock
        conv_block_1 = ConvBlock(in_dim, out_dim, k=k, s=s, p=p, 
                                 norm=True, non_linear='relu')
        conv_block_2 = ConvBlock(in_dim, out_dim, k=k, s=s, p=p, 
                                 norm=True, non_linear=None)
        self.res_block = nn.Sequential(conv_block_1, conv_block_2)
    
    def forward(self, x):
        out = x + self.res_block(x)
        return out
    
class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1, p=1):
        super(DeconvBlock, self).__init__()
        self.deconv_block = nn.Sequential(
                                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
                                nn.InstanceNorm2d(out_dim, affine=True),
                                nn.ReLU()
                            )
        
    def forward(self, x):
        out = self.deconv_block(x)
        return out
        
class Discriminator(nn.Module):
    def __init__(self, class_num=5):
        super(Discriminator, self).__init__()
        self.input_layer = ConvBlock(3, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu')
        self.hidden_layer = nn.Sequential(
                                ConvBlock(64, 128, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                ConvBlock(128, 256, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                ConvBlock(256, 512, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                ConvBlock(512, 1024, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                ConvBlock(1024, 2048, k=4, s=2, p=1, norm=False, non_linear='leaky_relu')
                            )
        self.src_layer = ConvBlock(2048, 1, k=3, s=1, p=1, norm=False, non_linear=None)
        self.cls_layer = ConvBlock(2048, class_num, k=2, s=1, p=0, norm=False, non_linear=None)
        
    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        src_score = self.src_layer(out)
        cls_score = self.cls_layer(out).squeeze()
        
        if cls_score.dim() == 1:
            cls_score = cls_score.unsqueeze(0)
        
        return src_score, cls_score
                                    
class Generator(nn.Module):
    def __init__(self, res_num=6, class_num=5):
        super(Generator, self).__init__()
        self.class_num = class_num
        
        self.down_sampling = nn.Sequential(
                                ConvBlock(3 + class_num, 64, k=7, s=1, p=3, norm=True, non_linear='relu'),
                                ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='relu'),
                                ConvBlock(128, 256, k=4, s=2, p=1, norm=True, non_linear='relu')
                             )
        
        bottle_neck = []
        for _ in range(res_num):
            bottle_neck += [ResBlock(256, 256, k=3, s=1, p=1)]
        self.bottle_neck = nn.Sequential(* bottle_neck)
        
        self.up_sampling = nn.Sequential(
                                DeconvBlock(256, 128, k=4, s=2, p=1),
                                DeconvBlock(128, 64, k=4, s=2, p=1),
                                ConvBlock(64, 3, k=7, s=1, p=3, norm=False, non_linear='tanh')
                           )
        
    def forward(self, x, label):
        # Reshape label and concat with x
        N, _, H, W = x.size()
        c = label.view(N, self.class_num, 1, 1).repeat(1, 1, H, W)
        x_with_c = torch.cat((x, c), dim=1)
        
        out = self.down_sampling(x_with_c)
        out = self.bottle_neck(out)
        out = self.up_sampling(out)
        
        return out