import torch
import torch.nn as nn
def double_conv(in_channels, out_channels):
    '''
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the 
    output result size to be same as input size.
    '''
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_channels, eps=1e-05,  momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(inplace = True, negative_slope = 0.01),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-05,  momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(inplace = True, negative_slope = 0.01),
    )
    return conv_op
class UNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, init_features = 96):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
         # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_conv(in_channels, init_features)
        self.down_convolution_2 = double_conv(init_features, init_features*2)
        self.down_convolution_3 = double_conv(init_features*2, init_features*4)
        self.down_convolution_4 = double_conv(init_features*4, init_features*8)
        self.down_convolution_5 = double_conv(init_features*8, init_features*16)
        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=init_features*16, out_channels=init_features*8, kernel_size=2, stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_conv(init_features*16, init_features*8)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=init_features*8, out_channels=init_features*4, kernel_size=2, stride=2)
        self.up_convolution_2 = double_conv(init_features*8, init_features*4)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=init_features*4, out_channels=init_features*2, kernel_size=2, stride=2)
        self.up_convolution_3 = double_conv(init_features*4, init_features*2)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=init_features*2, out_channels=init_features, kernel_size=2, stride=2)
        self.up_convolution_4 = double_conv(init_features*2, init_features)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(in_channels=init_features, out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)
        
        up_1 = self.up_transpose_1(down_9)        
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        
        x = self.out(x)
        
        return x
