import keras
from train_from_notebook import unet, create_dataset
import matplotlib.pyplot as plt

path_to_model = 'D:/Person/Drozdov/logs/_epoch-0020.h5'
uk = keras.models.load_model(path_to_model)
print(uk.summary())
holo_path = 'D:/Person/Drozdov/HOLO_25_inline/'
qr_path = 'D:/Person/Drozdov/QR_25_inline/'
train_dataset, val_dataset = create_dataset(qr_path, holo_path, crop=True)
fig, ax = plt.subplots(1, 3, figsize=(10, 8))
i = 3
for holo, qr in val_dataset:
    ax[0].imshow(holo[i, :, :, 0])
    ax[0].set_title('Синтезированная голограмма')
    ax[1].imshow(uk.predict(holo)[i, :, :, 0])
    ax[1].set_title('Восстановленное изображение')
    ax[2].imshow(qr[i, :, :, 0])
    ax[2].set_title('Реальное изображение')
    plt.show()
    break