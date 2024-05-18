from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Conv2DTranspose

def double_conv(filters_num1, filters_num2):
    conv_op = keras.Sequential(
        layers= [
        Conv2D(filters = filters_num1, kernel_size=32, padding = 'same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(filters = filters_num2, kernel_size=32, padding = 'same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
                ]   
    )
    return conv_op

class unet_keras(keras.Model):
    def __init__(self):
        super().__init__()
        self.maxpool2d = MaxPooling2D(pool_size = (2,2), strides = 2)

        self.down_convolution_1 = double_conv(filters_num1 = 16, filters_num2 = 16)
        self.down_convolution_2 = double_conv(filters_num1 = 32, filters_num2 = 32)
        self.down_convolution_3 = double_conv(filters_num1 = 64, filters_num2 = 64)
        self.down_convolution_4 = double_conv(filters_num1 = 128, filters_num2 = 128)
        self.down_convolution_5 = double_conv(filters_num1 = 256, filters_num2 = 256)
        self.down_convolution_6 = double_conv(filters_num1 = 512, filters_num2 = 512)
        
        self.up_transpose_1 = Conv2DTranspose(filters =  512, kernel_size= 2, strides= (2, 2))
        self.up_convolution_1 = double_conv(filters_num1 = 512, filters_num2 = 512)
        self.up_transpose_2 = Conv2DTranspose(filters =  256, kernel_size= 2, strides= (2, 2))
        self.up_convolution_2 = double_conv(filters_num1 = 256, filters_num2 = 256)
        self.up_transpose_3 = Conv2DTranspose(filters =  128, kernel_size= 2, strides= (2, 2))
        self.up_convolution_3 =  double_conv(filters_num1 = 128, filters_num2 = 128)
        self.up_transpose_4 = Conv2DTranspose(filters =  64, kernel_size= 2, strides= (2, 2))
        self.up_convolution_4 =  double_conv(filters_num1 = 64, filters_num2 = 64)

        self.out = Conv2D(filters = 1, kernel_size= 1)

    def call(self, inputs, training = False):
        x = self.down_convolution_1(inputs)
        x = self.maxpool2d(x)
        x = self.down_convolution_2(x)
        x = self.maxpool2d(x)
        x = self.down_convolution_3(x)
        x = self.maxpool2d(x)
        x = self.down_convolution_4(x)
        x = self.maxpool2d(x)
        # x = self.down_convolution_5(x)
        # x = self.maxpool2d(x)
        # x = self.down_convolution_6(x)
        x = self.out(x)
        return x
my_unet = unet_keras()


        
    



