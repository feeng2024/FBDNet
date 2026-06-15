import torch
import torch.nn as nn
from Net.PVTv2 import *
from Net.p2t import Encoder_p2t_base
from Net.p2t import Encoder_p2t_tiny
from Net.p2t import Encoder_p2t_small
from Net.p2t import Encoder_p2t_large
import torch.nn.functional as F
import timm
from SAFM import SAFM
from IBIM import IBIM
from T_Loss import TFM_loss
from FBAM import CA_Block_B
from FBAM import CA_Block_F
from FBAM import SA_Block_B
from FBAM import SA_Block_F


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class MBIM(nn.Module):
    def __init__(self, in_channel):
        super(MBIM,self).__init__()
        self.conv_l = ConvBR(in_channel, in_channel, 3, stride=1,padding=1)
        self.conv_m = ConvBR(in_channel, in_channel, 3, stride=1,padding=1)
        self.conv_s = ConvBR(in_channel, in_channel, 3, stride=1,padding=1)
        self.conv1 = ConvBR(in_channel*3, in_channel*3, 3, stride=1,padding=1)
        self.conv2 = ConvBR(in_channel*3, in_channel, 3, stride=1,padding=1)
        self.conv3 = ConvBR(in_channel, 1, 3, stride=1,padding=1)

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(192, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x_l,x_m,x_s,image_size):
        # x_l:17x17 x_m:11x11 x_s:6x6

        # x_l:Bx64x17x17
        x_l = self.conv_l(x_l) # Bx64x17x17
        x_l = F.interpolate(x_l, image_size, mode='bilinear', align_corners=False)
        x_l = self.conv_l(x_l) # Bx64x11x11

        # x_m:Bx64x11x11
        x_m = self.conv_m(x_m)
        x_m = F.interpolate(x_m, image_size, mode='bilinear', align_corners=False)
        x_m = self.conv_l(x_m)  # Bx64x11x11

        # x_s:Bx64x11x11
        x_s = self.conv_s(x_s)  # Bx64x6x6
        x_s = F.interpolate(x_s, image_size, mode='bilinear', align_corners=False)
        x_s = self.conv_s(x_s)  # Bx64x11x11

        out_att = self.att(torch.cat((x_l, x_m, x_s), dim=1)) # Bx192x1x1
        out = self.conv1(torch.cat((x_l, x_m, x_s), dim=1)) # Bx192x11x11?
        out = out * out_att # # Bx192x11x11
        out = self.conv2(out) # Bx64x11x11

        pred = self.conv3(out)
        return out, pred



class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.channel = channel
        self.cab_f = CA_Block_F(self.channel)
        self.sab_f = SA_Block_F(self.channel)
        self.cab_b = CA_Block_B(self.channel)
        self.sab_b = SA_Block_B(self.channel)

        self.conv1 = ConvBR(channel, channel, 3, stride=1, padding=1)
        self.conv2 = ConvBR(channel, channel, 3, stride=1, padding=1)
        self.conv3 = ConvBR(channel, channel, 3, stride=1, padding=1)
        self.conv4 = ConvBR(channel, channel, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(2 * channel, channel, 3, stride=1, padding=1)####nn.Conv2d
        self.conv6 = nn.Conv2d(2 * channel, channel, 3, stride=1, padding=1)####nn.Conv2d
        self.conv7 = ConvBR(channel, channel, 3, stride=1, padding=1)
        self.conv8 = ConvBR(channel, channel, 3, stride=1, padding=1)
    def forward(self, cur_x, h_x, mask):
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)


        out_1_f = self.cab_f(cur_x, mask_d)
        out_2_f = self.sab_f(cur_x, mask_d)

        out_1_b = self.cab_b(cur_x, mask_d)
        out_2_b = self.sab_b(cur_x, mask_d)

        out_1_f = self.conv1(out_1_f * h_x)
        out_2_f = self.conv2(out_2_f * h_x)
        out_1_b = self.conv3(out_1_b * h_x)
        out_2_b = self.conv4(out_2_b * h_x)

        out_F = torch.cat([out_1_f, out_2_f], dim=1)
        out_B = torch.cat([out_1_b, out_2_b], dim=1)

        out_F = self.conv5(out_F)
        out_B = self.conv6(out_B)

        return out_F,out_B


class MyNet(nn.Module):
    def __init__(self, channel=32, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]):
        super(MyNet, self).__init__()
        channel = channel
        self.model_arc = arc
        if arc == 'PVTv2-B0':
            print('--> using PVTv2-B0 right now')
            self.context_encoder = pvt_v2_b0(pretrained=True)  # 加载预训练模型
            in_channel_list = [64, 160, 256]
        elif arc == 'PVTv2-B1':
            print('--> using PVTv2-B1 right now')
            self.context_encoder = pvt_v2_b1(pretrained=True)  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B2':
            print('--> using PVTv2-B2 right now')
            self.context_encoder = pvt_v2_b2(pretrained=True)  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B2-li':
            print('--> using PVTv2-B2-li right now')
            self.context_encoder = pvt_v2_b2_li(pretrained=True)  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        # 主干网络采用PVTv2-B4
        elif arc == 'PVTv2-B4':
            print('--> using PVTv2-B4 right now')
            self.context_encoder = pvt_v2_b4(pretrained=True)  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        elif arc == 'PVTv2-B5':
            print('--> using PVTv2-B5 right now')
            self.context_encoder = pvt_v2_b5(pretrained=True)  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        # 主干网络采用P2T
        elif arc == 'P2T-base':
            print('--> using P2T-base right now')
            self.context_encoder = Encoder_p2t_base()  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        elif arc == 'P2T-small':
            print('--> using P2T-small right now')
            self.context_encoder = Encoder_p2t_small()  # 加载预训练模型
            in_channel_list = [128, 320, 512]
        elif arc == 'P2T-tiny':
            print('--> using P2T-tiny right now')
            self.context_encoder = Encoder_p2t_tiny()  # 加载预训练模型
            in_channel_list = [96, 240, 384]
        elif arc == 'P2T-large':
            print('--> using P2T-large right now')
            self.context_encoder = Encoder_p2t_large()  # 加载预训练模型
            in_channel_list = [128, 320, 640]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        self.dr2 = DimensionalReduction(in_channel=channel, out_channel=64)
        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=64)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=64)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=64)
        self.CNN_encode1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.CNN_encode2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.CNN_encode3 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.CNN_encode4 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)

        self.upsample15 = nn.Upsample(scale_factor=1.5, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.mbim = MBIM(64)
        self.decode4 = Decoder(64)
        self.decode3 = Decoder(64)
        self.decode2 = Decoder(64)
        self.decode1 = Decoder(64)
        self.pred8 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred7 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pred1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.TFM_loss = TFM_loss()

        self.safm1 = SAFM(64,64)
        self.safm2 = SAFM(64,64)
        self.safm3 = SAFM(64,64)
        self.safm4 = SAFM(64,64)

        self.ibim = IBIM(64)

    def forward(self, x):

        if self.model_arc == 'PVTv2-B4' or self.model_arc == 'PVTv2-B5' or self.model_arc == 'PVTv2-B0':
            endpoints = self.context_encoder.extract_endpoints(x)
            x1 = endpoints['reduction_2']  # 2x64x88x88
            x2 = endpoints['reduction_3']  # 2x128x44x44
            x3 = endpoints['reduction_4']  # 2x320x22x22
            x4 = endpoints['reduction_5']  # 2x512x11x11
        elif self.model_arc == 'P2T-base' or self.model_arc == 'P2T-small' or self.model_arc == 'P2T-tiny' or self.model_arc == 'P2T-large':
            # large x2:8x64x88x88 x3:8x128x44x44 x4:8x320x22x22 x5:8x640x11x11
            x4, x3, x2, x1 = self.context_encoder(x)

        ######## backbone
        shape = x.size()[2:]
        xr1 = self.dr2(x1) # [2, 64, 96, 96] t1
        xr2 = self.dr3(x2) # [2, 64, 48, 48] t2
        xr3 = self.dr4(x3) # [2, 64, 24, 24] t3
        xr4 = self.dr5(x4) # [2, 64, 12, 12] t4

        ######## transformer
        xr1 = self.safm1(xr1)  # [2, 64, 96, 96] t1
        xr2 = self.safm2(xr2)  # [2, 64, 48, 48] t2
        xr3 = self.safm3(xr3)  # [2, 64, 24, 24] t3
        xr4 = self.safm4(xr4)  # [2, 64, 12, 12] t4


        ######## CNN
        o_x = x                    # 1.0 Bx3x352x352
        o_x15 = self.upsample15(x) # 1.5 Bx3x528x528
        o_x05 = self.upsample05(x) # 0.5 Bx3x176x176


        x_l_1 = self.CNN_encode1(o_x15)
        x_l_4 = self.ibim(x_l_1)  # Bx64x17x17

        x_m_1 = self.CNN_encode1(o_x)
        x_m_4 = self.ibim(x_m_1)  # Bx64x11x11

        x_s_1 = self.CNN_encode1(o_x05)
        x_s_4 = self.ibim(x_s_1)  # Bx64x6x6

        out_cnn, cnn_pred = self.mbim(x_l_4, x_m_4, x_s_4, xr4.shape[2:])  # Bx64x11x11

        #######     decoder
        d4, d4_B = self.decode4(out_cnn, xr4, cnn_pred)
        d4_B = F.interpolate(d4_B, size=xr3.size()[2:], mode='bilinear')
        d4 = F.interpolate(d4, size=xr3.size()[2:], mode='bilinear')
        p4_B = self.pred8(d4_B)
        p4 = self.pred7(d4)
        loss4 = self.TFM_loss(d4, p4, p4_B)

        d3, d3_B = self.decode3(d4, xr3, p4)
        d3_B = F.interpolate(d3_B, size=xr2.size()[2:], mode='bilinear')
        d3 = F.interpolate(d3, size=xr2.size()[2:], mode='bilinear')
        p3_B = self.pred6(d3_B)
        p3 = self.pred5(d3)
        loss3 = self.TFM_loss(d3, p3, p3_B)

        d2, d2_B = self.decode2(d3, xr2, p3)
        d2_B = F.interpolate(d2_B, size=xr1.size()[2:], mode='bilinear')
        d2 = F.interpolate(d2, size=xr1.size()[2:], mode='bilinear')
        p2_B = self.pred4(d2_B)
        p2 = self.pred3(d2)
        loss2 = self.TFM_loss(d2, p2, p2_B)

        d1, d1_B = self.decode1(d2, xr1, p2)
        p1_B = self.pred2(d1_B)
        p1 = self.pred1(d1)
        loss1 = self.TFM_loss(d1, p1, p1_B)
        final_loss = (2 ** -1) * loss1 + (2 ** -2) * loss2 + (2 ** -3) * loss3 + (2 ** -4) * loss4
        p1 = F.interpolate(p1, size=shape, mode='bilinear')
        p1_B = F.interpolate(p1_B, size=shape, mode='bilinear')

        p_cnn = F.interpolate(cnn_pred, size=shape, mode='bilinear')

        return p1, p_cnn,p1_B,final_loss


if __name__ == '__main__':
    net = timm.create_model(model_name="resnet18", pretrained=False, in_chans=3, features_only=True)
    print(net.default_cfg)
