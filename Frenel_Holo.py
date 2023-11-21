import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim


def FrenelDifr(object_field, z, ps = 5.5e-6, lmbd = 532e-9, Reconstruct = False):
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

def ssim_coef(image1, image2):
    # Предполагается, что изображения image1 и image2 имеют одинаковый размер и глубину цвета.
    # Приведение изображений к диапазону [0, 1]
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    # Вычисление SSIM
    ssim_score = ssim(image1, image2, data_range=image2.max() - image2.min(), multichannel=True)
    return ssim_score

def correlation_coef(image1, image2):
    # Преобразование изображений в одномерные массивы
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    # Вычисление среднего значения и отклонения для каждого изображения
    mean1 = np.mean(image1_flat)
    mean2 = np.mean(image2_flat)
    std1 = np.std(image1_flat)
    std2 = np.std(image2_flat)
    # Вычисление ковариации между изображениями
    covariance = np.cov(image1_flat, image2_flat)[0][1]
    # Вычисление коэффициента корреляции
    correlation = covariance / (std1 * std2)
    return correlation

def quantize_signal(signal, levels):
    # Нормализация сигнала в диапазоне от 1 до levels - 1
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    # Квантование сигнала
    quantized_signal = np.round(normalized_signal * (levels - 1))
    return quantized_signal.astype(int)

def FullReconstruct(object_field, z_reg, dx = 5.5e-6, p = 1, plotting = True):    
    hologram_field = FrenelDifr(object_field, z = z_reg)
    N = len(object_field)
    C = np.sqrt(p * np.mean(np.abs(hologram_field)**2)) # p - отношение средних по пикселям интенсивностей опорной и объектной волны
    reference_wave = C * np.ones((N, N), dtype=complex)
    hologram_field += reference_wave
    # Расчет интенсивности восстановленного изображения
    intensity = quantize_signal(np.abs(hologram_field) ** 2, 256)
    rec_holo = FrenelDifr(intensity, z = z_reg, Reconstruct = True)
    rec_image = quantize_signal((np.abs(rec_holo)**2), 256)
    #object_field = quantize_signal((np.abs(object_field)**2), 256)
    object_field = np.abs(object_field)**2
    
    '''
    Докрутка яркости для для поля изображения
     rec_image[0 : amplitude.shape[0], 0 : amplitude.shape[0]] = quantize_signal(
        rec_image[0 : amplitude.shape[0], 0 : amplitude.shape[0]], 256
    )
    print('Коэффициент корреляции: {}'.format(
        correlation_coef(object_field[0 : amplitude.shape[0],0 : amplitude.shape[0]],
                   rec_image[0 : amplitude.shape[0], 0 : amplitude.shape[0]])))
    print('Коэффициент SSIM: {}'.format(
        ssim_coef(object_field[0 : amplitude.shape[0],0 : amplitude.shape[0]],
                   rec_image[0 : amplitude.shape[0], 0 : amplitude.shape[0]])))
    '''
    
    if plotting:
        fig, ax = plt.subplots(1,3, figsize = (10,8))
        ax[0].imshow(np.abs(object_field))
        ax[0].set_title('Объектное поле', fontsize = 10)
        ax[1].imshow(intensity)
        ax[1].set_title('Синтезированная голограмма', fontsize = 10)
        ax[2].imshow(rec_image)
        ax[2].set_title('Восстановленное изображение', fontsize = 10)
        plt.show()
    
    return np.array((object_field, intensity, rec_image))

    