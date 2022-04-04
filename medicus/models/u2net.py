from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cat(
    *t: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    return torch.cat(t, dim=dim)


def upsample_like(
    input: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    return F.interpolate(
        input=input,
        size=target.shape[2:],
        mode='bilinear', 
        align_corners=True)


class U2NetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dirate: int = 1,
    ) -> None:
        super(U2NetConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dirate,
            dilation=dirate)
        self.batch_norm = nn.BatchNorm2d(
            out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class RSUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super(RSUBlock, self).__init__()
        self.in_conv = U2NetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels)

        down_conv = [
            U2NetConvLayer(
                in_channels=out_channels,
                out_channels=mid_channels)]
        for _ in range(num_layers - 2):
            down_conv.append(
                U2NetConvLayer(
                    in_channels=mid_channels,
                    out_channels=mid_channels))
        self.down_conv = nn.ModuleList(down_conv)

        max_pool = []
        for _ in range(num_layers - 2):
            max_pool.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                    ceil_mode=True))
        self.max_pool = nn.ModuleList(max_pool)
        
        self.mid_conv = U2NetConvLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            dirate=2)

        up_conv = []
        for _ in range(num_layers - 2):
            up_conv.append(
                U2NetConvLayer(
                    in_channels=2 * mid_channels,
                    out_channels=mid_channels))
        up_conv.append(
            U2NetConvLayer(
                in_channels=2 * mid_channels,
                out_channels=out_channels))
        self.up_conv = nn.ModuleList(up_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []

        for conv in [
            self.in_conv, 
            self.down_conv[0]
        ]:
            x = conv(x)
            xs.append(x)

        for pool, conv in zip(
            self.max_pool, 
            self.down_conv[1:]
        ):
            x = pool(x)
            x = conv(x)
            xs.append(x)

        x = self.mid_conv(x)
        xs = list(reversed(xs))

        for i, conv in enumerate(self.up_conv):
            x = cat(x, xs[i])
            x = conv(x)
            
            if i < len(self.up_conv) - 1:
                x = upsample_like(x, xs[i + 1])

        return x + xs[-1]



class RSUFBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int
    ) -> None:
        super(RSUFBlock, self).__init__()

        self.in_conv = U2NetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels)
        
        down_conv = []
        for i in range(num_layers):
            c_in = out_channels if i == 0 else mid_channels
            c_out = mid_channels

            down_conv.append(
                U2NetConvLayer(
                    in_channels=c_in,
                    out_channels=c_out,
                    dirate=2**i))
        self.down_conv = nn.ModuleList(down_conv)

        up_conv = []
        for i in reversed(range(num_layers - 1)):
            c_in = 2 * mid_channels
            c_out = out_channels if i == 0 else mid_channels

            up_conv.append(
                U2NetConvLayer(
                    in_channels=2 * mid_channels,
                    out_channels=mid_channels,
                    dirate=2**i))
        self.up_conv = nn.ModuleList(up_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []

        for conv in [self.in_conv, *self.down_conv[:-1]]:
            x = conv(x)
            xs.append(x)
        
        x = self.down_conv[-1](x)
        xs = list(reversed(xs))
        
        for conv, x_ in zip(self.up_conv, xs):
            x = cat(x, x_)
            x = conv(x)

        return x + xs[-1]


def channels_enc(i: int) -> Tuple[int]:
    c_in = 2**(5 + i)
    c_mid = 2**(4 + i)
    c_out = 2**(6 + i)

    return c_in, c_mid, c_out


def channels_dec(i: int) -> Tuple[int]:
    c_in = 2**(10 - i)
    c_mid = 2**(7 - i)
    c_out = 2**(8 - i)

    return c_in, c_mid, c_out


class U2Net(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1
    ) -> None:
        super(U2Net, self).__init__()

        encoder = []
        for i in range(6):
            c_in, c_mid, c_out = channels_enc(i)
            num_l = 7 - i

            block = RSUBlock

            if i == 0:
                c_in = in_channels
                c_mid = 2**(5 + i) 
            if i > 3:
                num_l = 4
                block = RSUFBlock
            if i == 5:
                c_in, c_mid, c_out = channels_enc(5)

            encoder.append(
                block(
                    in_channels=c_in,
                    mid_channels=c_mid,
                    out_channels=c_out,
                    num_layers=num_l))
        self.encoder = nn.ModuleList(encoder)

        max_pool = []
        for i in range(5):
            max_pool.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                    ceil_mode=True))
        self.max_pool = nn.ModuleList(max_pool)

        decoder = []
        for i in range(5):
            c_in, c_mid, c_out = channels_dec(i)
            num_l = 3 + i

            block = RSUBlock

            if i == 0:
                c_in, c_mid = 2**10, 2**8, 2**9
                num_l = 4
                block = RSUFBlock

            encoder.append(
                block(
                    in_channels=c_in,
                    mid_channels=c_mid,
                    out_channels=c_out,
                    num_layers=num_l))
        self.decoder = nn.ModuleList(decoder)
        
        side_conv = []
        for i in range(6):
            c_in = 2**(5 + i)

            if i == 0:
                c_in = 2**6
            if i == 5:
                c_in = 2**9
            
            side_conv.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1))
        self.side_conv = nn.ModuleList(side_conv)
        self.out_conv = nn.Conv2d(
            in_channels=6 * out_channels,
            out_channels=out_channels,
            kernel_size=1)
        
        #1: i, 5, 6
        #2: 6, 5, 7
        #3: 7, 6, 8
        #4: 8, 7, 9
        #5: 9, 8, 9
        #6: 9, 8, 9
        #5: 10, 8, 10
        #4: 10, 7,  8
        #3:  9, 6,  7
        #2:  8, 5,  6
        #1:  7, 4,  6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs_enc = []

        for block, pool in zip(
            self.encoder, 
            self.max_pool
        ):
            x = block(x)
            xs_enc.append(x)
            x = pool(x)
        x = self.encoder[-1](x)
        xs_enc.append(x)
        xs_enc = list(reversed(xs_enc))

        xs_dec = []
        for block, x_ in zip(
            self.decoder, 
            xs_enc
        ):
            x = upsample_like(x, x_)
            x = cat(x, x_)
            x = block(x)
            xs_dec.append(x)
        xs_dec = list(reversed(xs_dec))

        xs = []
        for conv, x_ in zip(
            self.side_conv,
            xs_dec
        ):
            x = conv(x_)

            if len(xs) > 0:
                x = upsample_like(x, xs[0])
            xs.append(x)

        x = cat(*xs)
        x = self.out_conv(x)

        return [torch.sigmoid(x) for x in [x, *xs]]
        

        
        

            

        
        

        
        
        
        

        
        




