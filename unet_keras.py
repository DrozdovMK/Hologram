from tensorflow import keras
from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, BatchNormalization,LeakyReLU, MaxPooling2D, Conv2DTranspose, Concatenate

def double_conv(filters_num1, filters_num2):
    conv_op = keras.Sequential(
        layers= [Conv2D(filters = filters_num1, kernel_size=3, padding = 'same', activation='relu'),
                BatchNormalization(),
                Conv2D(filters = filters_num2, kernel_size=3, padding = 'same', activation='relu'),
                BatchNormalization()
                ],
                
        name = 'double_conv_with_{}_and_{}_filters'.format(filters_num1, filters_num2)
    )
    return conv_op

class unet_keras(keras.Model):
    def __init__(self):
        super().__init__()
        self.averagepool2d = AveragePooling2D(pool_size = (2,2), strides = 2)
        self.dropout = Dropout(rate=0.1)
        self.down_convolution_1 = double_conv(filters_num1 = 64, filters_num2 = 64)
        self.down_convolution_2 = double_conv(filters_num1 = 128, filters_num2 = 128)
        # self.down_convolution_3 = double_conv(filters_num1 = 128, filters_num2 = 128)
        # self.down_convolution_4 = double_conv(filters_num1 = 128, filters_num2 = 128)
        # self.down_convolution_5 = double_conv(filters_num1 = 256, filters_num2 = 256)
        # self.down_convolution_6 = double_conv(filters_num1 = 512, filters_num2 = 512)
        
        self.up_transpose_1 = Conv2DTranspose(filters =  128, kernel_size= 2, strides= (2, 2))
        self.up_convolution_1 = double_conv(filters_num1 = 128, filters_num2 = 128)
        self.up_transpose_2 = Conv2DTranspose(filters =  64, kernel_size= 2, strides= (2, 2))
        self.up_convolution_2 = double_conv(filters_num1 = 64, filters_num2 = 64)
        # self.up_transpose_3 = Conv2DTranspose(filters =  128, kernel_size= 2, strides= (2, 2))
        # self.up_convolution_3 =  double_conv(filters_num1 = 128, filters_num2 = 128)
        # self.up_transpose_4 = Conv2DTranspose(filters =  64, kernel_size= 2, strides= (2, 2))
        # self.up_convolution_4 =  double_conv(filters_num1 = 64, filters_num2 = 64)

        self.out = Conv2D(filters = 1, kernel_size= 1, activation= 'sigmoid')

    def call(self, inputs, training = False):

        x1 = self.down_convolution_1(inputs)
        pool1 = self.averagepool2d(x1)
        x2 = self.down_convolution_2(pool1)
        pool2 = self.averagepool2d(x2)

        up1 = self.up_transpose_1(pool2)
        x3 = self.up_convolution_1(Concatenate()([x2, up1]))
        up2 = self.up_transpose_2(x3)
        x4 = self.up_convolution_2(Concatenate()([x1, up2]))
        x5 = self.dropout(x4)
        # x = self.down_convolution_3(x)
        # x = self.maxpool2d(x)
        # x = self.down_convolution_4(x)
        # x = self.maxpool2d(x)
        # x = self.down_convolution_5(x)
        # x = self.maxpool2d(x)
        # x = self.down_convolution_6(x)
        x = self.out(x5)
        return x
my_unet = unet_keras()


        
    



