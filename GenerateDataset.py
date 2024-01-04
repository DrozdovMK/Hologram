import numpy as np
from Frenel_Holo import FullReconstruct
import cv2
import os
import qrcode

def create_qr(savepath, n_samples = 60, array_length = 10, version = 5,
              error_correction = qrcode.constants.ERROR_CORRECT_H, box_size = 1, border = 1):
    for n in range(n_samples):
        data = np.random.randint(10, size=array_length)
        qr = qrcode.QRCode(version = version, error_correction = error_correction, box_size = box_size, border = border)
        qr.add_data(data)
        qr.make(fit = True)
        img = qr.make_image(fill_color = 'black', back_color = 'white')
        img = 255*np.array(img, dtype=int)
        name = np.array2string(data) + '.png'
        cv2.imwrite(savepath + name, img)
    
def generate_dataset(path_to_images, write_path_holo, write_path_freconstr, N = 64, ps = 5.5e-6, wavelength = 532e-9):
    z_0 = N * ps * ps / wavelength
    for name in os.listdir(path_to_images):
        image = cv2.imread(path_to_images + name, cv2.IMREAD_GRAYSCALE)
        amplitude = np.sqrt(np.array(image))
        phase = np.random.uniform(0, 2*np.pi, amplitude.shape)
        object_field = np.zeros((N, N), dtype = complex)
        object_field[0: amplitude.shape[0], 0: amplitude.shape[0]] = amplitude * np.exp(1j * phase)
        registr_distantion = np.random.choice([1,3,6,10])
        object_field, hologram, rec_image = FullReconstruct(object_field, registr_distantion*z_0, plotting= False)
        cv2.imwrite(write_path_holo + name, hologram)
        cv2.imwrite(write_path_freconstr + name, rec_image)

SAVEPATH_QR = '../images/qr_for_seq/'
WRITEPATH_HOLO = '../images/holo_qr_for_seq/'
WRITEPATH_FREN = '../images/fren_qr_for_seq/'

create_qr(SAVEPATH_QR, n_samples= 60000)
generate_dataset(SAVEPATH_QR, WRITEPATH_HOLO, WRITEPATH_FREN)