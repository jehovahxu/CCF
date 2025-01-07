import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from scipy.signal import convolve2d
from math import exp

import torch
import torch.nn.functional as F

def rgb2gray(rgb):
    """
    convert rgb image into gray
    Args:
        rgb: grb image with numpy type
    Returns:
        gray: gray image
    """
    gray = rgb[:, 0:1, ...] * 0.299 + rgb[:, 1:2, ...] * 0.587 + rgb[:, 2:3, ...] * 0.114
    return gray


class Sobel():
    def __init__(self, devices):
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')

        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        self.weight = Variable(torch.from_numpy(sobel_kernel), requires_grad=True).to(devices)

        sobel_kernel_v = np.array([[-1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')

        sobel_kernel_v = sobel_kernel_v.reshape((1, 1, 3, 3))
        self.weight_v = Variable(torch.from_numpy(sobel_kernel_v)).to(devices)


        sobel_kernel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')

        sobel_kernel_h = sobel_kernel_h.reshape((1, 1, 3, 3))
        self.weight_h = Variable(torch.from_numpy(sobel_kernel_h)).to(devices)


    def sobel(self, im):
        import pdb;pdb.set_trace()
        if im.shape[1] == 3:
            im = rgb2gray(im)
        im = (im + 1) * 127.5
        edge_detect = F.conv2d(im, self.weight, padding=1)
        edge_detect = torch.clip(edge_detect, 0, 255)
        edge_detect = edge_detect/127.5 - 1

        return edge_detect

    def sobel_v(self, im):
        if im.shape[1] == 3:
            im = rgb2gray(im)
        im = (im + 1) * 127.5
        edge_detect = F.conv2d(Variable(im), self.weight_v, padding=1)
        edge_detect = torch.clip(edge_detect, 0, 255)
        edge_detect = edge_detect/127.5 - 1
        return edge_detect

    def sobel_h(self, im):
        # import pdb;pdb.set_trace()
        if im.shape[1] == 3:
            im = rgb2gray(im)
        im = (im + 1) * 127.5
        edge_detect = F.conv2d(Variable(im), self.weight_h, padding=1)
        edge_detect = torch.clip(edge_detect, 0, 255)
        edge_detect = edge_detect / 127.5 - 1
        return edge_detect

    def sobel(self, im):
        if im.shape[1] == 3:
            im = rgb2gray(im)
        im = (im + 1) * 127.5
        edge_detect = F.conv2d(im, self.weight, padding=1)
        edge_detect = torch.clip(edge_detect, 0, 255)
        edge_detect = edge_detect / 127.5 - 1

        return edge_detect


def AG(img):  # Average gradient
    # import pdb;pdb.set_trace()
    # img = rgb2gray((img+1)/2 * 255)
    Gx, Gy = torch.zeros_like(img), torch.zeros_like(img)

    Gx[..., :, 0] = img[..., :, 1] - img[..., :, 0]
    Gx[..., :, -1] = img[..., :, -1] - img[..., :, -2]
    Gx[..., :, 1:-1] = (img[..., :, 2:] - img[..., :, :-2]) / 2

    Gy[..., 0, :] = img[..., 1, :] - img[..., 0, :]
    Gy[:,:, -1, :] = img[..., -1, :] - img[..., -2, :]
    Gy[:,:, 1:-1, :] = (img[..., 2:, :] - img[..., :-2, :]) / 2
    ag = torch.mean(torch.sqrt((Gx ** 2 + Gy ** 2) / 2))
    # if torch.isnan(ag):
    #     import pdb;pdb.set_trace()
    # print("ag: %s", ag)
    return ag


def EI(img, sobel):  # Average gradient
    ### different sobel kernel has different results
    ei = 0.
    if img.shape[1] == 3:
        img = rgb2gray(img)
    edgesh = sobel.sobel_h(img)
    edgesv = sobel.sobel_v(img)
    ei = torch.mean(torch.sqrt(edgesh ** 2 + edgesv ** 2))
    return ei / img.shape[1]


def CE(image_F, image_A, image_B):

    histogram, bins = torch.histogram(image_F, bins=256, range=(0, 255))
    histogram_f = histogram / float(torch.sum(histogram))

    histogram, bins = torch.histogram(image_A, bins=256, range=(0, 255))
    histogram_a = histogram / float(torch.sum(histogram))

    histogram, bins = torch.histogram(image_B, bins=256, range=(0, 255))
    histogram_b = histogram / float(torch.sum(histogram))


    ce1 = torch.sum(histogram_a * torch.log2(histogram_f + 1e-7))
    ce2 = torch.sum(histogram_b * torch.log2(histogram_f + 1e-7))

    return torch.sqrt((ce1**2 + ce2**2) / 2)


def SF(img):
    sf = 0.
    # import pdb;pdb.set_trace()
    if img.shape[1] == 3:
        img = rgb2gray(img)
    # import pdb;pdb.set_trace()
    for i in range(img.shape[1]):
        sf += torch.sqrt(
            torch.sqrt(torch.mean((img[:, i, 1:, :] - img[:, i, :-1, :]) ** 2)) **2
          + torch.sqrt(torch.mean((img[:, i, :, 1:] - img[:, i, :, :-1]) ** 2)) **2
                                  )

    return sf / img.shape[1]


def CC(img1, img2):
    # cls.input_check(image_F, image_A, image_B)
    # cc = 0.
    # for i in range(img1.shape[1]):
    # import pdb;pdb.set_trace()
    rAF = torch.sum(
        (img1 - torch.mean(img1)) *
        (img2 - torch.mean(img2))) / torch.sqrt(
        (torch.sum((img1 - torch.mean(img1)) ** 2)) * (torch.sum((img2 - torch.mean(img2)) ** 2)))
    cc = rAF
    return cc

def SCD(img1, img2): # The sum of the correlations of differences
    # cls.input_check(image_F, image_A, image_B)
    imgF_A = img1 - img2
    scd = 0.
    for i in range(imgF_A.shape[1]):
        corr1 = torch.sum((img1[:, i] - torch.mean(img1[:, i])) * (imgF_A[:, i] - torch.mean(imgF_A[:, i]))) / torch.sqrt(
            (torch.sum((img1[:, i] - torch.mean(img1[:, i])) ** 2)) * (torch.sum((imgF_A[:, i] - torch.mean(imgF_A[:, i])) ** 2)))
        scd += corr1
    return scd

def SD(img):
    # some used all dimensions, some used ever dimension to cal deviation
    sd = 0.
    # import pdb;pdb.set_trace()
    for i in range(img.shape[1]):
        sd += torch.std(img[:,i])
    return sd / img.shape[1]

def VIFF(img1, img2):
    # cls.input_check(image_F, image_A, image_B)
    viff = 0.
    # tmp = img1.clone()
    # import pdb;pdb.set_trace()
    # img1 = rgb2gray(img1)
    # img2 = rgb2gray(img2)
    # import pdb;pdb.set_trace()
    for i in range(img1.shape[1]):
        viff += compare_viff(((img1[:, i,...] + 1) / 2) * 255, ((img2[:, i,...] + 1) / 2) * 255)
    return viff


def compare_viff(ref, dist): # viff of a pair of pictures
    sigma_nsq = 2.0
    eps = 1e-10

    num = 0.0
    den = 0.0
    scale = 1
    N = 2 ** (4 - scale + 1) + 1
    sd = N / 5.0

    # Create a Gaussian kernel similar to MATLAB's
    m, n = [(ss - 1.) / 2. for ss in (N, N)]
    y, x = torch.meshgrid([torch.arange(-m, m + 1), torch.arange(-n, n + 1)])
    h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        win = h / sumh
    # import pdb;pdb.set_trace()
    win = win.unsqueeze(0).unsqueeze(0).to(ref.device)  # Make it a 4D tensor for convolution


    mu1_1 = F.conv2d(ref.unsqueeze(0), win, padding=0)
    mu2_1 = F.conv2d(dist.unsqueeze(0), win, padding=0)
    mu1_sq_1 = mu1_1 * mu1_1
    mu2_sq_1 = mu2_1 * mu2_1
    mu1_mu2_1 = mu1_1 * mu2_1
    sigma1_sq_1 = F.conv2d((ref * ref).unsqueeze(0), win, padding=0) - mu1_sq_1
    sigma2_sq_1 = F.conv2d((dist * dist).unsqueeze(0), win, padding=0) - mu2_sq_1
    sigma12_1 = F.conv2d((ref * dist).unsqueeze(0), win, padding=0) - mu1_mu2_1

    sigma1_sq_1 = torch.clamp(sigma1_sq_1, min=0)
    sigma2_sq_1 = torch.clamp(sigma2_sq_1, min=0)

    g_1 = sigma12_1 / (sigma1_sq_1 + eps)
    sv_sq_1 = sigma2_sq_1 - g_1 * sigma12_1
    g_1_2 = torch.where(sigma1_sq_1 < eps, torch.tensor(0).to(g_1.device).float(),g_1)
    sv_sq_1[sigma1_sq_1 < eps] = sigma2_sq_1[sigma1_sq_1 < eps]
    sigma1_sq_1[sigma1_sq_1 < eps] = 0

    g_1_3 = torch.where(sigma2_sq_1 < eps, torch.tensor(0).to(g_1.device).float(),g_1_2)
    sv_sq_1[sigma2_sq_1 < eps] = 0

    sv_sq_1[g_1_3 < 0] = sigma2_sq_1[g_1_3 < 0]
    g_1_3 = torch.clamp(g_1_3, min=0)
    sv_sq_1 = torch.clamp(sv_sq_1, min=eps)

    num += torch.sum(torch.log10(1 + g_1_3 * g_1_3 * sigma1_sq_1 / (sv_sq_1 + sigma_nsq)))
    den += torch.sum(torch.log10(1 + sigma1_sq_1 / sigma_nsq))


    scale=2
    N = 2 ** (4 - scale + 1) + 1
    sd = N / 5.0

    # Create a Gaussian kernel similar to MATLAB's
    m, n = [(ss - 1.) / 2. for ss in (N, N)]
    y, x = torch.meshgrid([torch.arange(-m, m + 1), torch.arange(-n, n + 1)])
    h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        win = h / sumh
    # import pdb;pdb.set_trace()
    win = win.unsqueeze(0).unsqueeze(0).to(ref.device)  # Make it a 4D tensor for convolution

    ref_2 = F.conv2d(ref.unsqueeze(0), win, padding=0).squeeze(0)
    dist_2 = F.conv2d(dist.unsqueeze(0), win, padding=0).squeeze(0)
    ref_2 = ref_2[:, ::2, ::2]
    dist_2 = dist_2[:, ::2, ::2]

    mu1_2 = F.conv2d(ref_2.unsqueeze(0), win, padding=0)
    mu2_2 = F.conv2d(dist_2.unsqueeze(0), win, padding=0)
    mu1_sq_2 = mu1_2 * mu1_2
    mu2_sq_2 = mu2_2 * mu2_2
    mu1_mu2_2 = mu1_2 * mu2_2
    sigma1_sq_2 = F.conv2d((ref_2 * ref_2).unsqueeze(0), win, padding=0) - mu1_sq_2
    sigma2_sq_2 = F.conv2d((dist_2 * dist_2).unsqueeze(0), win, padding=0) - mu2_sq_2
    sigma12_2 = F.conv2d((ref_2 * dist_2).unsqueeze(0), win, padding=0) - mu1_mu2_2

    sigma1_sq_2 = torch.clamp(sigma1_sq_2, min=0)
    sigma2_sq_2 = torch.clamp(sigma2_sq_2, min=0)

    g_2 = sigma12_2 / (sigma1_sq_2 + eps)
    sv_sq_2 = sigma2_sq_2 - g_2 * sigma12_2

    # g_2[sigma1_sq_2 < eps] = 0
    g_2_2 = torch.where(sigma1_sq_2 < eps, torch.tensor(0).to(g_1.device).float(), g_2)
    sv_sq_2[sigma1_sq_2 < eps] = sigma2_sq_2[sigma1_sq_2 < eps]
    sigma1_sq_2[sigma1_sq_2 < eps] = 0

    # g_2[sigma2_sq_2 < eps] = 0
    g_2_3 = torch.where(sigma1_sq_2 < eps, torch.tensor(0).to(g_1.device).float(), g_2_2)
    sv_sq_2[sigma2_sq_2 < eps] = 0

    sv_sq_2[g_2_3 < 0] = sigma2_sq_2[g_2_3 < 0]
    g_2_3 = torch.clamp(g_2_3, min=0)
    sv_sq_2 = torch.clamp(sv_sq_2, min=eps)

    num += torch.sum(torch.log10(1 + g_2_3 * g_2_3 * sigma1_sq_2 / (sv_sq_2 + sigma_nsq)))
    den += torch.sum(torch.log10(1 + sigma1_sq_2 / sigma_nsq))






    scale = 3
    N = 2 ** (4 - scale + 1) + 1
    sd = N / 5.0

    # Create a Gaussian kernel similar to MATLAB's
    m, n = [(ss - 1.) / 2. for ss in (N, N)]
    y, x = torch.meshgrid([torch.arange(-m, m + 1), torch.arange(-n, n + 1)])
    h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        win = h / sumh
    # import pdb;pdb.set_trace()
    win = win.unsqueeze(0).unsqueeze(0).to(ref.device)  # Make it a 4D tensor for convolution

    ref_3 = F.conv2d(ref_2.unsqueeze(0), win, padding=0).squeeze(0)
    dist_3 = F.conv2d(dist_2.unsqueeze(0), win, padding=0).squeeze(0)
    ref_3 = ref_3[:, ::2, ::2]
    dist_3 = dist_3[:, ::2, ::2]

    mu1_3 = F.conv2d(ref_3.unsqueeze(0), win, padding=0)
    mu2_3 = F.conv2d(dist_3.unsqueeze(0), win, padding=0)
    mu1_sq_3 = mu1_3 * mu1_3
    mu2_sq_3 = mu2_3 * mu2_3
    mu1_mu2_3 = mu1_3 * mu2_3
    sigma1_sq_3 = F.conv2d((ref_3 * ref_3).unsqueeze(0), win, padding=0) - mu1_sq_3
    sigma2_sq_3 = F.conv2d((dist_3 * dist_3).unsqueeze(0), win, padding=0) - mu2_sq_3
    sigma12_3 = F.conv2d((ref_3 * dist_3).unsqueeze(0), win, padding=0) - mu1_mu2_3

    sigma1_sq_3 = torch.clamp(sigma1_sq_3, min=0)
    sigma2_sq_3 = torch.clamp(sigma2_sq_3, min=0)

    g_3 = sigma12_3 / (sigma1_sq_3 + eps)
    sv_sq_3 = sigma2_sq_3 - g_3 * sigma12_3

    # g_2[sigma1_sq_2 < eps] = 0
    g_3_2 = torch.where(sigma1_sq_3 < eps, torch.tensor(0).to(g_1.device).float(), g_3)
    sv_sq_3[sigma1_sq_3 < eps] = sigma2_sq_3[sigma1_sq_3 < eps]
    sigma1_sq_3[sigma1_sq_3 < eps] = 0

    # g_2[sigma2_sq_2 < eps] = 0
    g_3_3 = torch.where(sigma1_sq_3 < eps, torch.tensor(0).to(g_1.device).float(), g_3_2)
    sv_sq_3[sigma2_sq_3 < eps] = 0

    sv_sq_3[g_3_3 < 0] = sigma2_sq_3[g_3_3 < 0]
    g_3_3 = torch.clamp(g_3_3, min=0)
    sv_sq_3 = torch.clamp(sv_sq_3, min=eps)

    num += torch.sum(torch.log10(1 + g_3_3 * g_3_3 * sigma1_sq_3 / (sv_sq_3 + sigma_nsq)))
    den += torch.sum(torch.log10(1 + sigma1_sq_3 / sigma_nsq))



    scale = 4
    N = 2 ** (4 - scale + 1) + 1
    sd = N / 5.0

    # Create a Gaussian kernel similar to MATLAB's
    m, n = [(ss - 1.) / 2. for ss in (N, N)]
    y, x = torch.meshgrid([torch.arange(-m, m + 1), torch.arange(-n, n + 1)])
    h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        win = h / sumh
    # import pdb;pdb.set_trace()
    win = win.unsqueeze(0).unsqueeze(0).to(ref.device)  # Make it a 4D tensor for convolution

    ref_4 = F.conv2d(ref_3.unsqueeze(0), win, padding=0).squeeze(0)
    dist_4 = F.conv2d(dist_3.unsqueeze(0), win, padding=0).squeeze(0)
    ref_4 = ref_4[:, ::2, ::2]
    dist_4 = dist_4[:, ::2, ::2]

    mu1_4 = F.conv2d(ref_4.unsqueeze(0), win, padding=0)
    mu2_4 = F.conv2d(dist_4.unsqueeze(0), win, padding=0)
    mu1_sq_4 = mu1_4 * mu1_4
    mu2_sq_4 = mu2_4 * mu2_4
    mu1_mu2_4 = mu1_4 * mu2_4
    sigma1_sq_4 = F.conv2d((ref_4 * ref_4).unsqueeze(0), win, padding=0) - mu1_sq_4
    sigma2_sq_4 = F.conv2d((dist_4 * dist_4).unsqueeze(0), win, padding=0) - mu2_sq_4
    sigma12_4 = F.conv2d((ref_4 * dist_4).unsqueeze(0), win, padding=0) - mu1_mu2_4

    sigma1_sq_4 = torch.clamp(sigma1_sq_4, min=0)
    sigma2_sq_4 = torch.clamp(sigma2_sq_4, min=0)

    g_4 = sigma12_4 / (sigma1_sq_4 + eps)
    sv_sq_4 = sigma2_sq_4 - g_4 * sigma12_4

    # g_2[sigma1_sq_2 < eps] = 0
    g_4_2 = torch.where(sigma1_sq_4 < eps, torch.tensor(0).to(g_1.device).float(), g_4)
    sv_sq_4[sigma1_sq_4 < eps] = sigma2_sq_4[sigma1_sq_4 < eps]
    sigma1_sq_4[sigma1_sq_4 < eps] = 0

    # g_2[sigma2_sq_2 < eps] = 0
    g_4_3 = torch.where(sigma1_sq_4 < eps, torch.tensor(0).to(g_1.device).float(), g_4_2)
    sv_sq_4[sigma2_sq_4 < eps] = 0

    sv_sq_4[g_4_3 < 0] = sigma2_sq_4[g_4_3 < 0]
    g_4_3 = torch.clamp(g_4_3, min=0)
    sv_sq_4 = torch.clamp(sv_sq_4, min=eps)

    num += torch.sum(torch.log10(1 + g_4_3 * g_4_3 * sigma1_sq_4 / (sv_sq_4 + sigma_nsq)))
    den += torch.sum(torch.log10(1 + sigma1_sq_4 / sigma_nsq))


    vifp = num / den

    if torch.isnan(vifp):
        return torch.tensor(1).to(vifp.device)
    else:
        return vifp


def Qabf(image_F, image_A, image_B):
    # 检查输入的形状，如果是3通道图像，转换为灰度图
    if image_F.shape[0] == 3:
        image_F = torch.round(rgb2gray(image_F))
        image_A = torch.round(rgb2gray(image_A))
        image_B = torch.round(rgb2gray(image_B))
    else:
        image_F = image_F[:, 0]
        image_A = image_A[:, 0]
        image_B = image_B[:, 0]

    gA, aA = Qabf_getArray(image_A)
    gB, aB = Qabf_getArray(image_B)
    gF, aF = Qabf_getArray(image_F)
    QAF = Qabf_getQabf(aA, gA, aF, gF)
    QBF = Qabf_getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = torch.sum(gA + gB)
    nume = torch.sum(QAF * gA + QBF * gB)
    return nume / deno

def Qabf_getArray(img):
    # Sobel算子
    img = (img +1 ) /2 * 255
    h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device)
    h2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32, device=img.device)
    h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device)

    # 使用F.conv2d进行卷积运算
    SAx = F.conv2d(img.unsqueeze(0), h3.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)
    SAy = F.conv2d(img.unsqueeze(0), h1.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)
    gA = torch.sqrt(SAx * SAx + SAy * SAy)
    aA = torch.zeros_like(img, device=img.device)
    aA[SAx == 0] = math.pi / 2
    aA[SAx != 0] = torch.atan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA

def Qabf_getQabf(aA, gA, aF, gF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    GAF = torch.zeros_like(aA, device=aA.device)
    AAF = torch.zeros_like(aA, device=aA.device)
    QgAF = torch.zeros_like(aA, device=aA.device)
    QaAF = torch.zeros_like(aA, device=aA.device)
    QAF = torch.zeros_like(aA, device=aA.device)

    GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]

    AAF = 1 - torch.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF

    return QAF

if __name__ == '__main__':
    device = 'cuda:0'
    img1 = torch.randn((1,3,256,256)).to(device)
    img2 = torch.randn((1,3,256,256)).to(device)
    sobel = Sobel(device)
    # ei = EI(img1, sobel)

    cc = SCD(img1, img2)


