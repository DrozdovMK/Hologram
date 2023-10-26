import cv2
import matplotlib.pyplot as plt
import numpy as np
import qrcode
import os

def QR_Creating(readpath, writepath, qr_version = 17):
    for name in os.listdir(readpath):
        qr = qrcode.QRCode(version = qr_version, error_correction = qrcode.constants.ERROR_CORRECT_L, box_size = 1, border = 4)
        data = cv2.imread(readpath + name)
        qr.add_data(data)
        qr.make(fit = True)
        img = qr.make_image(fill_color = 'black', back_color = 'white')
        img = np.array(img, dtype = np.uint8)*255
        cv2.imwrite(writepath + name, img)

readpath = r'..\\images\\archive\\barrel_jellyfish\\'
writepath = r'..\\images\\qr_barrel_jellyfish\\'
QR_Creating(readpath, writepath, qr_version = 17)