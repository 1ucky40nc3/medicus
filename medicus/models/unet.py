import torch
import torch.nn as nn


class UNetConvLayer(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels,
            kernel_size=kernel_size,
            padding=padding)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        return x


class UNet(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        down_conv = []
        for i in range(6, 10):
            c_in = 2**(i - 1)
            c_out = 2**i

            if i == 6:
                c_in = in_channels

            self.down_conv.append(
                UNetConvLayer(c_in, c_out))
        self.down_conv = nn.ModuleList(down_conv)
        
        up_conv = []
        for i in reversed(range(7, 10)):
            c_in = 2**(i - 1) + 2**i
            c_out = 2**(i - 1)

            self.up_conv.append(
                UNetConvLayer(c_in, c_out))
        self.up_conv = nn.ModuleList(up_conv)

        self.last_conv = nn.Conv2d(
            in_channels=self.up_conv[-1].out_channels,
            out_channels=num_classes, 
            kernel_size=1)

        self.max_pool = nn.MaxPool2d(
            kernel_size=2)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []
        for down_conv in self.down_conv[:-1]:
            x = down_conv(x)
            xs.append(x)
            x = self.max_pool(x)
        x = self.down_conv[-1](x)

        for up_conv, x_ in zip(self.up_conv, reversed(xs)):
            x = self.upsample(x)
            x = torch.cat([x, x_], dim=1)
            x = up_conv(x)
        
        return self.last_conv(x)