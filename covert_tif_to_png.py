from PIL import Image
import os

# Папка с изображениями в формате .tif
input_dir = '/home/drozdovmk/holo_data/25_inline/HOLO_25_inline_tif/'

# Папка для сохранения конвертированных изображений в формате .png
output_dir = '/home/drozdovmk/holo_data/25_inline/HOLO_25_inline/'

# Создание папки для сохранения конвертированных изображений, если она не существует
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Конвертация изображений из формата .tif в формат .png
for filename in os.listdir(input_dir):
    new_filename = str(int(os.path.splitext(filename)[0])) + '.png'
    if filename.endswith('.tif') and not os.path.exists(os.path.join(output_dir, new_filename)):
        img = Image.open(os.path.join(input_dir, filename))
        img.save(os.path.join(output_dir, new_filename))