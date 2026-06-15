import torch
import torch.nn as nn

class CA_Block_F(nn.Module):
    def __init__(self, in_dim, agent_num=49,qkv_bias=True,agent_bias=True):
        super(CA_Block_F, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=3,padding=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=3,padding=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3,padding=1)

        self.scale = (in_dim // 8) ** -0.5

        self.an_bias = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.na_bias = nn.Parameter(torch.zeros(1, 1, in_dim))

        self.ah_bias = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, 1, in_dim))

        self.ha_bias = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.wa_bias = nn.Parameter(torch.zeros(1, 1, in_dim))  

        nn.init.trunc_normal_(self.an_bias, std=0.02)
        nn.init.trunc_normal_(self.na_bias, std=0.02)
        nn.init.trunc_normal_(self.ah_bias, std=0.02)
        nn.init.trunc_normal_(self.aw_bias, std=0.02)
        nn.init.trunc_normal_(self.ha_bias, std=0.02)
        nn.init.trunc_normal_(self.wa_bias, std=0.02)

        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.dwc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, groups=in_dim)

    def forward(self, x, mask):

        B, C, H, W = x.size()
        N = H * W
        q_x = self.query_conv(x).reshape(B, C, N)
        k_x = self.key_conv(x).reshape(B, C, N)
        v_x = self.value_conv(x).reshape(B, C, N)

        x = x.reshape(B,C,N)
        mask = mask.view(B, -1, N)
        q_x = q_x * mask
        k_x = k_x * mask

        agent_tokens = x

        position_bias1 = self.an_bias.repeat(B, 1, C)
        position_bias2 = (self.ah_bias + self.aw_bias).repeat(B, 1, 1)
        position_bias = position_bias1 +position_bias2

        agent_energy = torch.bmm(agent_tokens, k_x.permute(0, 2, 1))
        agent_attention = self.softmax(agent_energy * self.scale + position_bias)
        agent_out = torch.bmm(agent_attention, v_x)


        agent_bias1 = self.na_bias.repeat(B, C, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).repeat(B, C, 1)
        agent_bias = agent_bias1 + agent_bias2

        channel_energy = torch.bmm(q_x, agent_tokens.transpose(-2, -1))
        channel_attention = self.softmax(channel_energy * self.scale+ agent_bias )
        channel_out = torch.bmm(channel_attention, agent_out)

        x_out =channel_out.reshape(B, C, H, W)
        v_x = v_x.view(B, C, H, W)
        x_out = x_out + self.gamma * self.dwc(v_x)


        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out
class CA_Block_B(nn.Module):
    def __init__(self, in_dim, agent_num=49,qkv_bias=True,agent_bias=True):
        super(CA_Block_B, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=3,padding=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=3,padding=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3,padding=1)

        self.scale = (in_dim // 8) ** -0.5


        self.an_bias = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.na_bias = nn.Parameter(torch.zeros(1, 1, in_dim))

        self.ah_bias = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, 1, in_dim))

        self.ha_bias = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.wa_bias = nn.Parameter(torch.zeros(1, 1, in_dim))

        nn.init.trunc_normal_(self.an_bias, std=0.02)
        nn.init.trunc_normal_(self.na_bias, std=0.02)
        nn.init.trunc_normal_(self.ah_bias, std=0.02)
        nn.init.trunc_normal_(self.aw_bias, std=0.02)
        nn.init.trunc_normal_(self.ha_bias, std=0.02)
        nn.init.trunc_normal_(self.wa_bias, std=0.02)

        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.dwc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, groups=in_dim)

    def forward(self, x, mask):

        B, C, H, W = x.size()
        N = H * W
        q_x = self.query_conv(x).reshape(B, C, N)
        k_x = self.key_conv(x).reshape(B, C, N)
        v_x = self.value_conv(x).reshape(B, C, N)

        x = x.reshape(B,C,N)
        mask = (1-mask).view(B, -1, N)
        q_x = q_x * mask
        k_x = k_x * mask

        agent_tokens = x

        position_bias1 = self.an_bias.repeat(B, 1, C)
        position_bias2 = (self.ah_bias + self.aw_bias).repeat(B, 1, 1)
        position_bias = position_bias1 +position_bias2

        agent_energy = torch.bmm(agent_tokens, k_x.permute(0, 2, 1))
        agent_attention = self.softmax(agent_energy * self.scale + position_bias)
        agent_out = torch.bmm(agent_attention, v_x)


        agent_bias1 = self.na_bias.repeat(B, C, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).repeat(B, C, 1)
        agent_bias = agent_bias1 + agent_bias2

        channel_energy = torch.bmm(q_x, agent_tokens.transpose(-2, -1))
        channel_attention = self.softmax(channel_energy * self.scale+ agent_bias )
        channel_out = torch.bmm(channel_attention, agent_out)

        x_out =channel_out.reshape(B, C, H, W)
        v_x = v_x.view(B, C, H, W)
        x_out = x_out + self.gamma * self.dwc(v_x)


        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out
class SA_Block_F(nn.Module):
    def __init__(self, in_dim, agent_num=49):
        super(SA_Block_F, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.agent_num = agent_num
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))


        self.an_bias = nn.Parameter(torch.zeros(1, agent_num, 1))
        self.na_bias = nn.Parameter(torch.zeros(1, 1, agent_num))

        self.ah_bias = nn.Parameter(torch.zeros(1, agent_num, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, 1, agent_num))


        self.ha_bias = nn.Parameter(torch.zeros(1, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, 1, agent_num))


        nn.init.trunc_normal_(self.an_bias, std=0.02)
        nn.init.trunc_normal_(self.na_bias, std=0.02)
        nn.init.trunc_normal_(self.ah_bias, std=0.02)
        nn.init.trunc_normal_(self.aw_bias, std=0.02)
        nn.init.trunc_normal_(self.ha_bias, std=0.02)
        nn.init.trunc_normal_(self.wa_bias, std=0.02)


        self.scale = (in_dim // 8) ** -0.5
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.dwc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, groups=in_dim)

    def forward(self, x, mask):
        B, C, H, W = x.size()
        N = H * W


        q_x = self.query_conv(x).view(B, -1, H * W)
        k_x = self.key_conv(x).view(B, -1, H * W)
        v_x = self.value_conv(x).view(B, -1, H * W)


        mask = mask.view(B, -1, N)
        q_x = q_x * mask
        k_x = k_x * mask

        agent_tokens = x

        position_bias1 = self.an_bias.repeat(B, 1, N)
        position_bias = position_bias1

        agent_attn = self.softmax((agent_tokens @ k_x) * self.scale + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v_x.permute(0, 2, 1)

        agent_bias1 = self.na_bias.repeat(B, N, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).repeat(B, N, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q_x.permute(0, 2, 1) @ agent_tokens.transpose(-2, -1)) * self.scale + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x_out = q_attn @ agent_v


        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        v_x = v_x.view(B, C, H, W)
        x_out = x_out + self.gamma*self.dwc(v_x)

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)

        return x_out
class SA_Block_B(nn.Module):
    def __init__(self, in_dim, agent_num=49):
        super(SA_Block_B, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.agent_num = agent_num
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))


        self.an_bias = nn.Parameter(torch.zeros(1, agent_num, 1))
        self.na_bias = nn.Parameter(torch.zeros(1, 1, agent_num))

        self.ah_bias = nn.Parameter(torch.zeros(1, agent_num, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, 1, agent_num))

        self.ha_bias = nn.Parameter(torch.zeros(1, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, 1, agent_num))

        nn.init.trunc_normal_(self.an_bias, std=0.02)
        nn.init.trunc_normal_(self.na_bias, std=0.02)
        nn.init.trunc_normal_(self.ah_bias, std=0.02)
        nn.init.trunc_normal_(self.aw_bias, std=0.02)
        nn.init.trunc_normal_(self.ha_bias, std=0.02)
        nn.init.trunc_normal_(self.wa_bias, std=0.02)


        self.scale = (in_dim // 8) ** -0.5
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.dwc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, groups=in_dim)  # 深度可分离卷积

    def forward(self, x, mask):
        B, C, H, W = x.size()
        N = H * W


        q_x = self.query_conv(x).view(B, -1, H * W)
        k_x = self.key_conv(x).view(B, -1, H * W)
        v_x = self.value_conv(x).view(B, -1, H * W)


        mask = (1-mask).view(B, -1, N)
        q_x = q_x * mask
        k_x = k_x * mask

        agent_tokens = x

        position_bias1 = self.an_bias.repeat(B, 1, N)
        position_bias = position_bias1

        agent_attn = self.softmax((agent_tokens @ k_x) * self.scale + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v_x.permute(0, 2, 1)


        agent_bias1 = self.na_bias.repeat(B, N, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).repeat(B, N, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q_x.permute(0, 2, 1) @ agent_tokens.transpose(-2, -1)) * self.scale + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x_out = q_attn @ agent_v


        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        v_x = v_x.view(B, C, H, W)
        x_out = x_out + self.gamma*self.dwc(v_x)


        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)

        return x_out
