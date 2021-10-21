from numpy.lib.arraypad import pad
import torch.nn as nn
import torch

class table_net(nn.Module):
    def __init__(self, num_classes):
        super(table_net, self).__init__()
        self.down1 = CNN_BLOCK(3, 16, downsampling=True)  # 256
        self.down2 = CNN_BLOCK(16, 32, downsampling=True)  # 128
        self.down3 = CNN_BLOCK(32, 64, downsampling=True)  # 64
        self.down4 = CNN_BLOCK(64, 128, downsampling=True)  # 32
        self.down5 = CNN_BLOCK(128, 256, downsampling=True)  # 16
        self.down6 = CNN_BLOCK(256, 512, downsampling=True)  # 8
        self.center = CNN_BLOCK(512, 1024)
        self.up1 = CNN_BLOCK(1024, 512, upsampling=True)  # 16
        self.up2 = CNN_BLOCK(512, 256, upsampling=True)  # 32
        self.up3 = CNN_BLOCK(256, 128, upsampling=True)  # 64
        self.up4 = CNN_BLOCK(128, 64, upsampling=True)  # 128
        self.up5 = CNN_BLOCK(64, 32, upsampling=True)  # 256
        self.up6 = CNN_BLOCK(32, 16, upsampling=True)  # 512

        self.classify = nn.Conv2d(16, num_classes, kernel_size=(1, 1))

    def forward(self, input):
        down1_pooling, down1 = self.down1(input)
        down2_pooling, down2 = self.down2(down1_pooling)
        down3_pooling, down3 = self.down3(down2_pooling)
        down4_pooling, down4 = self.down4(down3_pooling)
        down5_pooling, down5 = self.down5(down4_pooling)
        down6_pooling, down6 = self.down6(down5_pooling)
        center = self.center(down6_pooling)
        up6 = self.up1((center, down6))
        up5 = self.up2((up6, down5))
        up4 = self.up3((up5, down4))
        up3 = self.up4((up4, down3))
        up2 = self.up5((up3, down2))
        up1 = self.up6((up2, down1))
        output = torch.sigmoid(self.classify(up1))
        return output


class CNN_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3, 3), padding_mode = 'zeros', bias = False, downsampling = False, upsampling = False):
        super(CNN_BLOCK, self).__init__()

        if upsampling:
            self.up_sampling = nn.Upsample(scale_factor=2)
            in_channels = in_channels + out_channels
            self.cnn = _cnn_block(in_channels, out_channels, kernel_size, padding_mode, bias)
            in_channels = out_channels
        self.cnn1 = _cnn_block(in_channels, out_channels, kernel_size, padding_mode, bias)
        self.cnn2 = _cnn_block(out_channels, out_channels, kernel_size, padding_mode, bias)
        self.downsampling = downsampling
        self.upsampling = upsampling

        if downsampling:
            self.max_pooling = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, input):

        if self.upsampling:
            input, input_down = input
            input = self.up_sampling(input)
            input = torch.cat((input_down, input), dim = 1)
            input = self.cnn(input)

        output = self.cnn1(input)
        output = self.cnn2(output)

        if self.downsampling:
            output_pooling = self.max_pooling(output)
            return output_pooling, output

        return output

class _cnn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding_mode, bias):
        super(_cnn_block, self).__init__()
        self.CNN = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding_mode=padding_mode, bias=bias, padding=1)
        self.BatchNormal = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input):
        output = self.CNN(input)
        # _output = output
        output = self.BatchNormal(output)
        output = self.LeakyReLU(output)
        return output
