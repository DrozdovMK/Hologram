import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim


def FresnelDifr(object_field, z, ps = 5.5e-6, lmbd = 532e-9, Reconstruct = False):
    N = len(object_field)
    if not Reconstruct:
        du = ps
        dx_object = 5.5e-6
        dx_object = N * ps**3 / (lmbd * z)
        #print('Размер отсчета картинки', dx_object)
    else:
        dx_object = ps
        du = z * lmbd / (N * dx_object)
    # Cимметричная сетка
    x1 = np.linspace(-N//2 * dx_object, N//2 * dx_object, N) #Координата в поле объекта x
    y1 = np.linspace(-N//2 * dx_object, N//2 * dx_object, N) #Координата в поле объекта y
    u1 = np.linspace(-N//2 * du, N//2 * du, N) #Координата в поле восстановления u 
    v1 = np.linspace(-N//2 * du, N//2 * du, N) #Координата в поле восстановления v
    x, y = np.meshgrid(x1, y1)
    u, v = np.meshgrid(u1, v1)
    k = 2 * np.pi / lmbd
    B = ifftshift(ifft2(fftshift((object_field*np.exp(1j * np.pi/(lmbd * z)*(x**2 + y**2))))))
    B *= np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(
        1j * np.pi * (u**2 + v**2) / (lmbd * z))
    return B



def quantize_signal(signal, levels):
    'Перевод в диапазон от 0 до 255'
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    quantized_signal = np.round(normalized_signal * (levels - 1))
    return quantized_signal.astype(int)

def FullReconstruct(object_field, z_reg, ps = 5.5e-6, lmbd = 532e-9, p = 1, plotting = True):    
    hologram_field = FresnelDifr(object_field, z_reg, ps, lmbd, Reconstruct = False)
    N = len(object_field)
    C = np.sqrt(p * np.mean(np.abs(hologram_field)**2)) # p - отношение средних по пикселям интенсивностей опорной и объектной волны
    reference_wave = C * np.ones((N, N), dtype=complex)
    hologram_field += reference_wave
    hologram_abs = quantize_signal(np.abs(hologram_field) ** 2, 256) # Голограмма
    fren_rec = FresnelDifr(hologram_abs, z_reg, ps, lmbd, Reconstruct = True) 
    fren_rec = quantize_signal((np.abs(fren_rec)**2), 256) # Реконструкция изображения интегралом Френеля
    #object_field = quantize_signal((np.abs(object_field)**2), 256)
    object_field = np.abs(object_field)**2
    
    if plotting:
        fig, ax = plt.subplots(1,3, figsize = (10,8))
        ax[0].imshow(np.abs(object_field))
        ax[0].set_title('Объектное поле', fontsize = 10)
        ax[1].imshow(hologram_abs)
        ax[1].set_title('Синтезированная голограмма', fontsize = 10)
        ax[2].imshow(fren_rec)
        ax[2].set_title('Восстановленное изображение', fontsize = 10)
        plt.show()
    
    return np.array((object_field, hologram_abs, fren_rec))

    