import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import normalize
import scipy.misc
from matplotlib import pyplot as plt
import imageio
import cv2
import random
from PIL import Image
from PIL import Image
import numpy
import math
import tensorflow as tf
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim




# #先将matlab中的数据存为matl类型
# import scipy.io as
# matfn=u'E:\matlabDM\论文算法\isira\LLM.mat'
# data=sio.loadmat(matfn) #读取mat数据，转化为dict
# LM=data['L']; #读取字典中的稀疏矩阵，LM在python变量中


class KSVD(object):
    def load_data(file_path):
        # '''导入数据
        # input:  file_path(string):文件的存储位置
        # output: data(mat):数据
        # '''
        f = open(file_path)
        data = []
        for line in f.readlines():
            row = []  # 记录每一行
            lines = line.strip().split("\t")
            for x in lines:
                row.append(float(x))  # 将文本中的特征转换成浮点数
            data.append(row)
        f.close()
        return np.mat(data)  # 把data转换成matrix
    # 1e-6
    def __init__(self, n_components, max_iter=30, tol=5000,
                 n_nonzero_coefs=None):
        # """
        # 稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        # :param n_components: 字典所含原子个数（字典的列数）
        # :param max_iter: 最大迭代次数
        # :param tol: 稀疏表示结果的容差
        # :param n_nonzero_coefs: 稀疏度
        # """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        # """
        # 初始化字典矩阵
        # """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]
        print(self.dictionary.shape)

    def _update_dict(self, y, d, x):
        # """
        # 使用KSVD更新字典的过程
        # """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        # """
        # KSVD迭代过程
        # """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode








# if __name__ == '__main__':
#     im_ascent = scipy.misc.ascent().astype(np.float)
#     ksvd = KSVD(300)
#     dictionary, sparsecode = ksvd.fit(im_ascent)
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(im_ascent)
#     plt.subplot(1, 2, 2)
#     plt.imshow(dictionary.dot(sparsecode))
#     plt.show()
if __name__ == '__main__':
    # im_ascent = cv2.imread("./zaosheng1.png", 0).astype(np.float)
    im_ascent = cv2.imread("./zaosheng1.png")
    print(im_ascent.shape)
    img = im_ascent[:, :, (2, 1, 0)]
    # Gray = 0.299R+0.587G+0.114*B
    r, g, b = [img[:, :, i] for i in range(3)]
    img_gray = r * 0.299 + g * 0.587 + b * 0.114
    ksvd = KSVD(300)
    dictionary, sparsecode = ksvd.fit(img_gray)
    # cv2.imwrite("./input.png", img_gray.astype(np.uint8))
    output = dictionary.dot(sparsecode)
    print(output.shape)
    output = np.clip(output, 0, 255)
    cv2.imwrite("./output.png", output.astype(np.uint8))
    # # wuzao变成灰度图
    # im_wuzao = cv2.imread("./wuzao.png")
    # wuzao = im_wuzao[:, :, (2, 1, 0)]
    # # Gray = 0.299R+0.587G+0.114*B
    # r, g, b = [wuzao[:, :, i] for i in range(3)]
    # wuzao_gray = r * 0.299 + g * 0.587 + b * 0.114
    # cv2.imwrite("./wuzaohui.png", wuzao_gray.astype(np.uint8))
    img1 = cv2.imread("./wuzaohui.png")
    img2 = cv2.imread("./input.png")
    img3 = cv2.imread("./output.png")

    p = compare_psnr(img1, img2)
    p1 = compare_psnr(img1, img3)
    # s = compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
    m = compare_mse(img1, img2)
    m1 = compare_mse(img1, img3)
    # print('PSNR：{}，SSIM：{}，MSE：{}'.format(p, s, m))
    print('PSNR：{}，MSE：{}'.format(p, m))
    print('PSNR：{}，MSE：{}'.format(p1, m1))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap="gray")
    plt.axis('off')
    plt.show()


