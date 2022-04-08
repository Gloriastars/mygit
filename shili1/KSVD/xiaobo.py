import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import normalize
import scipy.misc
from matplotlib import pyplot as plt
import imageio
import cv2
import random
import math
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 中文显示工具函数
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    set_ch()
    img1 = cv2.imread("./wuzaohui.png")
    # original = img_as_float(data.img1())

    plt.figure()
    plt.subplot(221)
    plt.axis('off')
    plt.title('原始图像')
    plt.imshow(img1)
    # plt.imshow(original)

    # sigma = 0.2
    # noisy = random_noise(original, var=sigma ** 2)
    img2 = cv2.imread("./input.png")
    plt.subplot(222)
    plt.axis('off')
    plt.title('加噪图像')
    plt.imshow(img2)
    plt.show()
    # im_haar = denoise_wavelet(img2, wavelet='db2', multichannel=True, convert2ycbcr=True)
    im_haar = denoise_wavelet(img2, wavelet='db2')
    plt.figure(1, figsize=(800, 450))
    plt.axis('off')
    # plt.subplot(223)
    # plt.title('使用haar去噪后')
    plt.imshow(im_haar, cmap='gray')
    # figure(1, figsize=(450, 800), frameon=None)
    plt.savefig('./xiaoboquzao.png')
    plt.show()
    print(im_haar.shape)
    # cv2.imwrite("./xiaoboquzao.png", im_haar.astype(np.uint8))
    # 三通道变单通道
    # plt.figure(1)
    # img = Image.open('./xiaoboquzao.png')
    # xiaobo = np.array(img.convert('L'))
    # plt.imshow(xiaobo, cmap='gray')
    # plt.axis('off')
    # plt.savefig('./xiaobo.png')
    # print(xiaobo.shape)
    # plt.show()
# 不同颜色通道的噪声平均标准差
#     sigma_est = estimate_sigma(img2, multichannel=True, average_sigmas=True)
#     im_haar_sigma = denoise_wavelet(img2, wavelet='db2', multichannel=True, convert2ycbcr=True, sigma=sigma_est)
    sigma_est = estimate_sigma(img2)
    im_haar_sigma = denoise_wavelet(img2, wavelet='db2',  sigma=sigma_est)
    plt.figure(1)
    # plt.subplot(224)
    plt.axis('off')
    # plt.title('使用haar with sigma去噪后')
    plt.imshow(im_haar_sigma, cmap='gray')
    plt.axis('off')
    plt.savefig('./xiaoboquzao1.png')
    plt.show()


    # img1 = cv2.imread("./wuzaohui.png")
    # img2 = cv2.imread("./input.png")
    # img4 = cv2.imread("./output.png")
    #
    # p = compare_psnr(img1, img2)
    # p1 = compare_psnr(img1, img3)
    # # s = compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
    # m = compare_mse(img1, img2)
    # m1 = compare_mse(img1, img3)
    # # print('PSNR：{}，SSIM：{}，MSE：{}'.format(p, s, m))
    # print('PSNR：{}，MSE：{}'.format(p, m))
    # print('PSNR：{}，MSE：{}'.format(p1, m1))
