import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


class ContextualAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError('channels must be divisible by heads')
        self.channels = channels
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ProposedLiverSegNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, attention_heads=4, dropout=0.3):
        super().__init__()
        c = base_channels
        self.e1 = ConvBlock(in_channels, c, dropout=0.0)
        self.e2 = ConvBlock(c, c * 2, dropout=0.0)
        self.e3 = ConvBlock(c * 2, c * 4, dropout=0.1)
        self.e4 = ConvBlock(c * 4, c * 8, dropout=0.1)
        self.e5 = ConvBlock(c * 8, c * 16, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        self.context = ContextualAttention(c * 16, heads=attention_heads)

        self.d4 = UpBlock(c * 16, c * 8, c * 8, dropout=0.1)
        self.d3 = UpBlock(c * 8, c * 4, c * 4, dropout=0.1)
        self.d2 = UpBlock(c * 4, c * 2, c * 2, dropout=0.0)
        self.d1 = UpBlock(c * 2, c, c, dropout=0.0)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b = self.e5(self.pool(s4))
        b = self.context(b)
        x = self.d4(b, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.out(x)
