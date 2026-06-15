import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()
    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return 1 - sim

class TFM_loss(nn.Module):
    def __init__(self, indim=64):
        super(TFM_loss, self).__init__()


    def forward(self, input_F,input_P,input_B):#def forward(self, input_F,input_P,input_B):
        B, C, H ,W = input_F.shape
        in_F = input_F.mean(1,keepdim=True).reshape(B, -1)
        in_P = input_P.reshape(B, -1)
        in_B = input_B.reshape(B, -1)

        dis_neg = cos_distance(in_F,in_B)
        neg_pair = torch.diagonal(dis_neg, dim1=0, dim2=1)
        dis_pos = cos_distance(in_F, in_P)
        pos_pair = torch.diagonal(dis_pos, dim1=0, dim2=1)

        triple_loss = torch.relu(0.5 + pos_pair - neg_pair)

        res=triple_loss

        forepixel = input_F
        forepixel = forepixel.clone().reshape(B, -1)
        forepixel = forepixel.mean(1)
        loss = res + forepixel 
        loss = loss.mean(0)
        return loss
