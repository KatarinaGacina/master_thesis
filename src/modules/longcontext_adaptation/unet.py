import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channel,
                out_channel,
                stride=stride,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False, #bias=False because of BatchNorm1d
            ),
            nn.BatchNorm1d(out_channel), #nn.GroupNorm(num_groups=min(32, out_channel), num_channels=out_channel), or LayerNorm/RMSNorm with permute
            nn.GELU()
        )

        """self.conv = nn.Conv1d(
            in_channel,
            out_channel,
            stride=stride,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False, #bias=False because of BatchNorm1d
        )
        self.norm = nn.RMSNorm(out_channel, eps=1e-8)
        self.act = nn.GELU()"""

    def forward(self, x):
        """x = self.conv(x)
        x = x.transpose(1,2)
        x = self.norm(x)
        x = x.transpose(1,2)
        return self.act(x)"""
        return self.conv_block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()

        self.conv1 = Conv1dBlock(in_channel, out_channel, kernel_size=kernel_size, stride=1)
        self.conv2 = Conv1dBlock(out_channel, out_channel, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        res = self.conv1(x)
        out = self.conv2(res)

        return out, res

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, residual):
        super().__init__()

        self.residual = residual

        self.conv_transpose = nn.ConvTranspose1d(in_channel, in_channel, kernel_size=2, stride=2)
        self.act = nn.GELU()

        if (self.residual):
            self.conv = Conv1dBlock(2 * in_channel, out_channel, kernel_size=kernel_size, stride=1)
        else:
            self.conv = Conv1dBlock(in_channel, out_channel, kernel_size=kernel_size, stride=1)


    def forward(self, x, residual_value=None):
        if self.residual:
            assert residual_value is not None

            out = self.conv_transpose(x)
            assert (
                residual_value.shape == out.shape
            ), "residual-input shape missmatch"

            out = torch.concat([out, residual_value], dim=1)
            out = self.act(out)
            out = self.conv(out)
        
        else:
            out = self.conv_transpose(x)
            out = self.act(out)
            out = self.conv(out)

        return out


class DownsizeBlock(nn.Module):
    def __init__(self, num_downsampling, start_embed_dim, kernel_length=5, residual=True):
        super().__init__()

        self.residual = residual
        
        self.num_downsampling = num_downsampling
        self.downsize_layers = nn.ModuleList()

        self.embedding_factor = 128 #hardcoded, same as Enformer

        in_channel_dim = start_embed_dim
        out_channel_dim = in_channel_dim + self.embedding_factor

        for _ in range(num_downsampling):
            downsize_block = DownBlock(in_channel_dim, out_channel_dim, kernel_length)
            self.downsize_layers.append(downsize_block)

            in_channel_dim = out_channel_dim
            out_channel_dim = in_channel_dim + self.embedding_factor

    def forward(self, x):
        assert x.shape[2] // (2 ** self.num_downsampling) > 0; "sequence dimension too small"
        assert x.shape[2] % (2 ** self.num_downsampling) == 0; "sequence dimension is not even"

        residual_values = []

        for layer in self.downsize_layers:
            x, r = layer(x)
            residual_values.append(r)

        if self.residual:
            return x, residual_values
        else:
            return x

    def get_embedding_factor(self):
        return self.embedding_factor


class UpsizeBlock(nn.Module):
    def __init__(self, num_downsampling, current_embed_dim, kernel_length=5, residual=True):
        super().__init__()

        self.residual = residual

        self.upsize_layers = nn.ModuleList()

        self.embedding_growth_factor = 128 #hardcoded, same as Enformer

        in_channel_dim = current_embed_dim
        out_channel_dim = in_channel_dim - self.embedding_growth_factor

        for _ in range(num_downsampling):
            upsize_block = UpBlock(in_channel_dim, out_channel_dim, kernel_length, residual)
            self.upsize_layers.append(upsize_block)

            in_channel_dim = out_channel_dim
            out_channel_dim = in_channel_dim - self.embedding_growth_factor

    def forward(self, x, residual_values=None):
        if self.residual:
            assert residual_values is not None

            for layer, residual in zip(self.upsize_layers, reversed(residual_values)):
                x = layer(x, residual)
        else:
            for layer in self.upsize_layers:
                x = layer(x)

        return x
