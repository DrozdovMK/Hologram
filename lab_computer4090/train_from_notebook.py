# from unet_keras import unet_keras
import numpy as np
import tensorflow as tf
import os
import keras
from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, BatchNormalization,LeakyReLU, MaxPooling2D, Conv2DTranspose, Concatenate, UpSampling2D
import matplotlib.pyplot as plt
a = np.random.random(size = (1, 2048,2048,1))
holo_path = 'D:/Person/Drozdov/HOLO_25_inline/'
qr_path = 'D:/Person/Drozdov/QR_25_inline/'


def double_conv(filters_num1, filters_num2):
    conv_op = keras.Sequential(
        layers=[Conv2D(filters=filters_num1, kernel_size=4, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2D(filters=filters_num2, kernel_size=4, padding='same', activation='relu'),
                BatchNormalization()
                ],
    )
    return conv_op


def interleave(inputs):
    return tf.nn.space_to_depth(inputs, 16)


def deinterleave(inputs):
    return tf.nn.depth_to_space(inputs, 2)





def unet():
    inputs = Input(shape = [2048, 2048, 1])

    x0 = interleave(inputs)
    x1 = double_conv(filters_num1 = 64, filters_num2 = 64)(x0)
    pool1 = MaxPooling2D(pool_size = (2,2), strides = 2)(x1)  # 1024
    x2 = double_conv(filters_num1 = 128, filters_num2 = 128)(pool1)
    pool2 = MaxPooling2D(pool_size = (2,2), strides = 2)(x2)  # 512
    x3 = double_conv(filters_num1 = 256, filters_num2 = 256)(pool2)
    pool3 = MaxPooling2D(pool_size = (2,2), strides = 2)(x3)  # 256
    x4 = double_conv(filters_num1 = 512, filters_num2 = 512)(pool3)
    pool4 = MaxPooling2D(pool_size = (2,2), strides = 2)(x4)  # 128

    up1 = UpSampling2D(size=2)(pool4)  # 256
    x5 = double_conv(filters_num1 = 512, filters_num2 = 512)(Concatenate()([x4, up1]))
    up2 = UpSampling2D(size=2)(x5)  # 512
    x6 = double_conv(filters_num1 = 256, filters_num2 = 256)(Concatenate()([x3, up2]))
    up3 = UpSampling2D(size=2)(x6)  # 1024
    x7 = double_conv(filters_num1 = 128, filters_num2 = 128)(Concatenate()([x2, up3]))
    up4 = UpSampling2D(size=2)(x7)  # 2048
    x8 = double_conv(filters_num1 = 64, filters_num2 = 64)(Concatenate()([x1, up4]))

    x9 = Conv2D(filters = 4, kernel_size=1, activation = 'relu')(x8)
    x10 = deinterleave(x9)
    x11 = Conv2D(filters=4, kernel_size=57, strides=(1, 1), padding='valid', activation='relu')(x10)
    x12 = Conv2D(filters=1, kernel_size=1, activation='relu')(x11)
    x13 = AveragePooling2D(pool_size = (8,8), strides = 8, padding='valid')(x12)

    return tf.keras.Model(inputs=inputs, outputs=x13)

uk = unet()
# uk.build(a.shape)

random_seed = 44
def binarize_image(image, threshold = 30):
    binary_image = tf.where(image < threshold, 0, 1)
    return binary_image
def normalize_image(image):
    return tf.cast(image, tf.float32) / 255.0
def crop_image(image):
    cropped_image = tf.image.crop_to_bounding_box(image, 4, 4, 25, 25)  # Пример обрезки изображения
    return cropped_image
def combine_datasets(input_ds, output_ds):
    dataset = tf.data.Dataset.zip((input_ds, output_ds))
    return dataset
def create_dataset(qr_path, holo_path, crop = True):
    qr_train, qr_val = tf.keras.preprocessing.image_dataset_from_directory(qr_path, labels = None,
                                                    color_mode = 'grayscale', batch_size = 10, image_size = (34, 34),
                                                    subset = 'both', validation_split=0.02, seed = random_seed, interpolation='nearest')
    holo_train, holo_val = tf.keras.preprocessing.image_dataset_from_directory(holo_path, labels = None,
                                                    color_mode = 'grayscale', batch_size = 10, image_size = (2048, 2048),
                                                    subset = 'both', validation_split=0.02, seed = random_seed)

    # qr_train_binarized = qr_train.map(binarize_image, 30)
    # qr_val_binarized = qr_val.map(binarize_image, 30)

    qr_train_cropped = qr_train.map(crop_image)
    qr_val_cropped = qr_val.map(crop_image)
    train_dataset = combine_datasets(holo_train, qr_train_cropped)
    val_dataset = combine_datasets(holo_val, qr_val_cropped)
    return train_dataset, val_dataset
if __name__ == "__main__":
    train_dataset, val_dataset = create_dataset(qr_path, holo_path, crop=True)

    i = 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    for holo, qr in train_dataset:
        ax[0].imshow(qr[i, :, :, 0])
        ax[1].imshow(holo[i, :, :, 0])
        print(qr[i, :, :, 0].shape)
        plt.show()
        break

    # my_callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(filepath='D:/Person/Drozdov/model.{epoch:02d}-{val_loss:.2f}.h5' ),
    # ]

    checkpoint_path = 'D:/Person/Drozdov/logs/'


    class CustomCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            if (epoch % 1) == 0:
                self.model.save(checkpoint_path + '_epoch-{:04d}.h5'.format(epoch + 1))
                # self.model.save(checkpoint_path + '_lr-{:.5f}_epoch-{:04d}.h5'.format(lr, epoch+1))


    checkpoint_callback = CustomCheckpoint()

    uk.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
               loss='mse',
               metrics=['accuracy'],
               )
    history = uk.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=checkpoint_callback)

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    i = 3
    for holo, qr in train_dataset:
        i = 0
        ax[0].imshow(holo[i, :, :, 0])
        ax[0].set_title('Синтезированная голограмма')
        ax[1].imshow(uk.predict(holo)[i, :, :, 0])
        ax[1].set_title('Восстановленное изображение')
        ax[2].imshow(qr[i, :, :, 0])
        ax[2].set_title('Реальное изображение')
        plt.show()
        break

    uk.save('D:/Person/Drozdov/unet_model')