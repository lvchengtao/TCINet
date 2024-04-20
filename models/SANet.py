import torch
from torch import nn
from torch.nn import functional as F

from models.swinNet import SwinTransformer


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return x.mul(torch.sigmoid(out))


class SpatialAttention_wo_sig(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return x






class BasicConv2d_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y

class Cross_3d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.cro_mod_2d= BasicConv2d_relu(2,1,3,1,1)
        self.cro_mod_2d2= BasicConv2d_relu(2,1,3,1,1)
        self.spat_r= SpatialAttention_wo_sig()
        self.spat_t= SpatialAttention_wo_sig()
        self.cor= CoordAtt(in_channels,in_channels)
        self.conv_fus= BasicConv2d_relu(in_channels*2,in_channels,3,1,1)
    def forward(self, E_rgb5_enh, E_fs5_enh):
        E_rgb5_enh_s= self.spat_r(E_rgb5_enh)
        E_fs5_enh_s= self.spat_t(E_fs5_enh)
        corre_spatia_5= torch.cat([E_rgb5_enh_s,E_fs5_enh_s],dim=1)
        corre_spatia_5_1= torch.sigmoid(self.cro_mod_2d(corre_spatia_5))
        corre_spatia_5_2= torch.sigmoid(self.cro_mod_2d2(corre_spatia_5))
        corre_spatia_5= torch.cat([E_rgb5_enh* corre_spatia_5_1, E_fs5_enh*corre_spatia_5_2],dim=1)
        corre_spatia_5= self.conv_fus(corre_spatia_5)
        corre_spatia_5= self.cor(corre_spatia_5)


        return corre_spatia_5

class Decoder_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_1= BasicConv2d_relu(in_channels*2,in_channels,3,1,1)
        self.conv_2= BasicConv2d_relu(in_channels*2,in_channels,3,1,1)
        self.dec_fus4= BasicConv2d_relu(in_channels*2,in_channels,3,1,1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down2= nn.AvgPool2d(2,2)
    def forward(self, f4, f5):

        f5_2= self.conv_1(torch.cat([self.down2(f4), f5],dim=1))+ f5
        f4_2= self.conv_2(torch.cat([f4, self.upsample2(f5)],dim=1))+ f4

        f4= self.dec_fus4(torch.cat([f4_2,self.upsample2(f5_2)],dim=1))

        return f4

class SANet(nn.Module):
  def __init__(self):
    super(SANet, self).__init__()

    channel_decoder = 32
    # encoder
    self.rgb_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])

    self.rfb5= BasicConv2d_relu(512*2,channel_decoder,3,1,1)
    self.rfb4= BasicConv2d_relu(256*2,channel_decoder,3,1,1)
    self.rfb3= BasicConv2d_relu(128*2,channel_decoder,3,1,1)
    self.rfb2= BasicConv2d_relu(64*2,channel_decoder,3,1,1)

    self.rfb5_t= BasicConv2d_relu(512*2,channel_decoder,3,1,1)
    self.rfb4_t= BasicConv2d_relu(256*2,channel_decoder,3,1,1)
    self.rfb3_t= BasicConv2d_relu(128*2,channel_decoder,3,1,1)
    self.rfb2_t= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    
    self.cro_mod_5= Cross_3d(channel_decoder)
    self.cro_mod_4= Cross_3d(channel_decoder)
    self.cro_mod_3= Cross_3d(channel_decoder)
    self.cro_mod_2= Cross_3d(channel_decoder)

    self.dec_fus_r4= Decoder_2(channel_decoder)
    self.dec_fus_r3= Decoder_2(channel_decoder)
    self.dec_fus_r2= Decoder_2(channel_decoder)

    self.dec_fus_t4= Decoder_2(channel_decoder)
    self.dec_fus_t3= Decoder_2(channel_decoder)
    self.dec_fus_t2= Decoder_2(channel_decoder)
    

    self.conv_pre_r2= nn.Conv2d(channel_decoder,1,1,1,0)
    self.conv_pre_t2= nn.Conv2d(channel_decoder,1,1,1,0)
    
    self.conv_pre_f5= nn.Conv2d(channel_decoder,1,1,1,0)
    self.conv_pre_f4= nn.Conv2d(channel_decoder,1,1,1,0)
    self.conv_pre_f3= nn.Conv2d(channel_decoder,1,1,1,0)
    self.conv_pre_f2= nn.Conv2d(channel_decoder,1,1,1,0)


    self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)


  def load_pre(self, pre_model):
    self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
    print(f"RGB SwinTransformer loading pre_model ${pre_model}")

  def forward(self, rgb, fss):

    E_rgb1, E_rgb2, E_rgb3, E_rgb4, E_rgb5 = self.rgb_swin(rgb)
    E_fs1, E_fs2, E_fs3, E_fs4, E_fs5 = self.rgb_swin(fss)

    E_rgb5= self.rfb5(E_rgb5)
    E_fs5= self.rfb5_t(E_fs5)
    E_fs5_enh= E_fs5
    E_rgb5_enh= E_rgb5
    f5= self.cro_mod_5(E_rgb5_enh,E_fs5_enh)
    
    E_rgb4= self.dec_fus_r4(self.rfb4(E_rgb4),f5)
    E_fs4= self.dec_fus_t4(self.rfb4_t(E_fs4),f5)
    E_fs4_enh= E_fs4
    E_rgb4_enh= E_rgb4
    f4= self.cro_mod_4(E_rgb4_enh,E_fs4_enh)

    E_rgb3= self.dec_fus_r3(self.rfb3(E_rgb3),f4)
    E_fs3= self.dec_fus_t3(self.rfb3_t(E_fs3),f4)
    E_fs3_enh= E_fs3
    E_rgb3_enh= E_rgb3
    f3= self.cro_mod_3(E_rgb3_enh,E_fs3_enh)

    E_rgb2= self.dec_fus_r2(self.rfb2(E_rgb2),f3)
    E_fs2= self.dec_fus_t2(self.rfb2_t(E_fs2),f3)

    E_fs2_enh= E_fs2
    E_rgb2_enh= E_rgb2
    
    f2= self.cro_mod_2(E_rgb2_enh,E_fs2_enh)
    
    pre_f5= self.conv_pre_f5(f5)
    
    pre_f4= self.conv_pre_f4(f4)
    
    pre_f3= self.conv_pre_f3(f3)

    xr2= self.conv_pre_r2(E_rgb2)
    xf2= self.conv_pre_t2(E_fs2)
    
    x2_pre= self.conv_pre_f2(f2)

    pre_f1_three= self.upsample4(xr2+xf2+x2_pre)

    return pre_f1_three,self.upsample4(x2_pre),self.upsample4(xr2),self.upsample4(xf2),self.upsample8(pre_f3),self.upsample16(pre_f4),self.upsample32(pre_f5)


  def initialize_weights(self):
    pass

if __name__ == '__main__':
    from thop import profile
    a= SANet()
    input1 = torch.randn(1, 3, 384, 384)
    input2 = torch.randn(1, 3, 384, 384)
    macs, params = profile(a, inputs=(input1,input2))
    print(macs,params)