from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, BatchNormalization,LeakyReLU, MaxPooling2D, Conv2DTranspose, Concatenate, UpSampling2D
import tensorflow as tf
import numpy as np
def double_conv(filters_num1, filters_num2):
    conv_op = keras.Sequential(
        layers= [Conv2D(filters = filters_num1, kernel_size=4, padding = 'same', activation='relu'),
                BatchNormalization(),
                Conv2D(filters = filters_num2, kernel_size=4, padding = 'same', activation='relu'),
                BatchNormalization()
                ],
                
        name = 'double_conv_with_{}_and_{}_filters'.format(filters_num1, filters_num2)
    )
    return conv_op

def interleave(inputs):
    return tf.nn.space_to_depth(inputs, 16)


def deinterleave(inputs):
    return tf.nn.depth_to_space(inputs, 2)


class unet_keras(keras.Model):
    def __init__(self):
        super().__init__()
        self.maxpool2d = MaxPooling2D(pool_size = (2,2), strides = 2)
        self.averagepool2d = AveragePooling2D(pool_size = (8,8), strides = 8, padding='valid')
        # self.dropout = Dropout(rate=0.1)

        self.interleave = interleave
        self.deinterleave = deinterleave
        self.down_convolution_1 = double_conv(filters_num1 = 64, filters_num2 = 64)
        self.down_convolution_2 = double_conv(filters_num1 = 128, filters_num2 = 128)
        self.down_convolution_3 = double_conv(filters_num1 = 256, filters_num2 = 256)
        self.down_convolution_4 = double_conv(filters_num1 = 512, filters_num2 = 512)

        self.up_sample1 = UpSampling2D(size=2)
        self.up_convolution_1 = double_conv(filters_num1 = 512, filters_num2 = 512)
        self.up_sample2 = UpSampling2D(size=2)
        self.up_convolution_2 = double_conv(filters_num1 = 256, filters_num2 = 256)
        self.up_sample3 = UpSampling2D(size=2)
        self.up_convolution_3 = double_conv(filters_num1=128, filters_num2=128)
        self.up_sample4 = UpSampling2D(size=2)
        self.up_convolution_4 = double_conv(filters_num1=64, filters_num2=64)
        self.conv_same = Conv2D(filters = 4, kernel_size=1, activation = 'relu')


        # self.up_transpose_last = Conv2DTranspose(filters = 1, kernel_size= 9, strides= (1, 1), padding = 'valid', activation='relu')
        self.convolution_prelast = Conv2D(filters=4, kernel_size=57, strides=(1, 1), padding='valid', activation='relu')
        self.conv_last = Conv2D(filters=1, kernel_size=1, activation='relu')
        # self.up_convolution_3 =  double_conv(filters_num1 = 128, filters_num2 = 128)
        # self.up_transpose_4 = Conv2DTranspose(filters =  64, kernel_size= 2, strides= (2, 2))
        # self.up_convolution_4 =  double_conv(filters_num1 = 64, filters_num2 = 64)

        # self.out = Conv2D(filters = 1, kernel_size= 1, activation= 'sigmoid')

    def call(self, inputs, training = False):

        x00 = inputs
        # x0 = self.(x00)
        x1 = self.down_convolution_1(x0)
        pool1 = self.maxpool2d(x1) #1024
        x2 = self.down_convolution_2(pool1)
        pool2 = self.maxpool2d(x2) #512
        x3 = self.down_convolution_3(pool2)
        pool3 = self.maxpool2d(x3) #256
        x4 = self.down_convolution_4(pool3)
        pool4 = self.maxpool2d(x4) #128


        up1 = self.up_sample1(pool4) #256
        x5 = self.up_convolution_1(Concatenate()([x4, up1]))
        up2 = self.up_sample2(x5) #512
        x6 = self.up_convolution_2(Concatenate()([x3, up2]))
        up3 = self.up_sample2(x6) #1024
        x7 = self.up_convolution_3(Concatenate()([x2, up3]))
        up4 = self.up_sample2(x7) #2048
        x8 = self.up_convolution_4(Concatenate()([x1, up4]))

        x9 = self.conv_same(x8)
        x10 = self.deinterleave(x9)
        x11 = self.convolution_prelast(x10)
        x12 = self.conv_last(x11)
        x13 = self.averagepool2d(x12)
        return x13
a = np.random.random((1,2048,2048,1))
my_unet = unet_keras()
print(my_unet(a))


        
    



