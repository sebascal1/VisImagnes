#Original author jpcano1, all content belongs to him. Used under authorization#

import torch
from torch import nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels
        :param padding: The padding size
        :param args: Function arguments
        :param kwargs: Function Keyword arguments
        """
        super(ConvBlock, self).__init__()
        # The kernel size
        kernel_size = kwargs.get("kernel_size") or 3
        # The stride size
        stride = kwargs.get("stride") or 1

        # Padding Mode
        padding_mode = kwargs.get("padding_mode") or "zeros"
        # Activation Function
        activation = kwargs.get("activation") or nn.LeakyReLU(0.2)
        # Batch Normalization
        bn = kwargs.get("bn") or False

        layers = []

        # Convolutional layer creation
        conv2d_layer = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding_mode=padding_mode, 
                                padding=padding)
        layers.append(conv2d_layer)
        if bn:
            bn_layer = nn.BatchNorm2d(out_channels)
            layers.append(bn_layer)

        layers.append(activation)
        # The creation of the layers from a Sequential module.
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        return self.conv_block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2, mode="bilinear", 
                 *args, **kwargs):
        """
        Initializer method
        :param scale_factor: The factor of upsampling
        :param mode: The mode of upsampling,
        generally an interpolation method
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(UpsampleBlock, self).__init__()

        # Conditional modes
        if mode != "nearest":
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor, 
                                              mode=mode, align_corners=True)
        else:
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor,
                                              mode=mode)
            
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be
        :return: The tensor forwarded to the
        upsample block
        """
        return self.upsample_layer(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels.
        :param args: Function arguments.
        :param kwargs: Function keyword arguments.
        """
        super(DownBlock, self).__init__()

        # Create the layers list
        layers = []

        # The number of convolutions per block
        jump = kwargs.get("jump") or 1

        # The initial layer of the jump loop
        init_layer = ConvBlock(in_channels, out_channels, 
                               *args, **kwargs)
        
        layers.append(init_layer)

        # Append convolutional blocks if
        # jump is greater than 1
        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, 
                              *args, **kwargs)
            layers.append(layer)

        self.down_block = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to down block
        """
        return self.down_block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(UpBlock, self).__init__()

        jump = kwargs.get("jump") or 1
        layers = []

        # The upsampling layer
        self.upsample = UpsampleBlock(*args, **kwargs)

        # The first convolutional layer
        layer = ConvBlock(in_channels, out_channels, 
                          *args, **kwargs)
        layers.append(layer)

        # Append convolutional blocks
        # if jump is greater than 1
        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, 
                              *args, **kwargs)
            layers.append(layer)
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, down_block, last_block):
        """
        The forward method
        :param down_block: The tensor from the down block
        :param last_block: The tensor from the last block
        :return:
        """
        last_block = self.upsample(last_block)
        x = torch.cat((last_block, down_block), dim=1)
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, init_filters, depth, *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param init_filters: The number of init filters
        :param depth: The depth of the Net
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(Encoder, self).__init__()
        layers = []
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2

        # The number of convolutions per block
        jump = kwargs.get("jump") or 1

        # First layer
        init_layer = ConvBlock(in_channels, init_filters, 
                               *args, **kwargs)
        layers.append(init_layer)

        current_filters = init_filters

        # Loop down through the depth
        for _ in range(depth - 1):
            # Append convolutional blocks
            # if jump is greater than 1
            for _ in range(jump - 1):
                # Convolution Block
                layer = ConvBlock(current_filters,
                                  current_filters * 2, 
                                  *args,
                                  **kwargs)
                # Append Layers
                layers.append(layer)

                current_filters *= 2

            # Pooling Block
            layer = nn.MaxPool2d(kernel_size=pool_size, 
                                 stride=pool_stride)
            layers.append(layer)

            # Convolution Block
            layer = ConvBlock(current_filters,
                              current_filters * 2, 
                              *args, **kwargs)
            layers.append(layer)

            current_filters *= 2

        # Create encoder
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The forwarded tensor
        """
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, init_filters, out_channels, depth, *args, **kwargs):
        """
        Initializer Method
        :param init_filters:
        :param out_channels:
        :param depth:
        :param kwargs:
        """
        super(Decoder, self).__init__()
        layers = []

        scale_factor = kwargs.get("scale_factor") or 2 
        mode = kwargs.get("mode") or "bilinear"

        jump = kwargs.get("jump") or 1

        current_filters = init_filters

        for _ in range(depth - 1):    
            for _ in range(jump - 1):
                layer = ConvBlock(current_filters,
                                  current_filters // 2,
                                  *args, **kwargs)
                layers.append(layer)

                current_filters //= 2

            layer = UpsampleBlock(scale_factor=scale_factor, mode=mode)
            layers.append(layer)
            layer = ConvBlock(current_filters,
                              current_filters // 2, 
                              *args, **kwargs)
            layers.append(layer)

            current_filters //= 2

        layer = ConvBlock(current_filters, out_channels, 
                          activation=nn.Sigmoid(), *args, **kwargs)
        layers.append(layer)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        
        :param x:
        :type x:
        :return:
        :rtype:
        """
        return self.decoder(x)
