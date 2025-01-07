import os
import numpy as np
import cv2
import torch


class Color_np():
    @staticmethod
    def YCbCr2RGB(Y, Cb, Cr):
        """
        将YcrCb格式转换为RGB格式
        :param Y:
        :param Cb:
        :param Cr:
        :return:
        """
        ycrcb = np.cancatenate([Y, Cr, Cb], dim=1)
        C, W, H = ycrcb.shape
        im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
        mat = np.array([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]])
        bias = np.array([0.0 / 255, -0.5, -0.5])
        temp = (im_flat + bias).mm(mat)
        out = temp.reshape(W, H, C).transpose(0, 2).transpose(1, 2)
        out = np.clip(out, 0, 1.0)
        return out
    @staticmethod
    def RGB2YCrCb(rgb_image):
        """
        将RGB格式转换为YCrCb格式
        用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
        :param rgb_image: RGB格式的图像数据
        :return: Y, Cr, Cb
        """

        R = rgb_image[0:1]
        G = rgb_image[1:2]
        B = rgb_image[2:3]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5

        Y = np.clip(Y,0.0,1.0)
        Cr = np.clip(Cr,0.0,1.0)
        Cb = np.clip(Cb, 0.0,1.0)
        return Y, Cb, Cr

class Color_tensor():
    @staticmethod
    def RGB2YCrCb(rgb_image):
        """
        将RGB格式转换为YCrCb格式
        用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
        :param rgb_image: RGB格式的图像数据
        :return: Y, Cr, Cb
        """

        R = rgb_image[:, 0:1]
        G = rgb_image[:, 1:2]
        B = rgb_image[:, 2:3]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5

        # Y = Y.clamp(0.0, 1.0)
        # Cr = Cr.clamp(0.0, 1.0).detach()
        # Cb = Cb.clamp(0.0, 1.0).detach()
        return Y, Cb, Cr

    @staticmethod
    def YCbCr2RGB(Y, Cb, Cr):
        """
        将YcrCb格式转换为RGB格式
        :param Y:
        :param Cb:
        :param Cr:
        :return:
        """
        ycrcb = torch.cat([Y, Cr, Cb], dim=1)
        B, C, W, H = ycrcb.shape
        im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
        mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
                           ).to(Y.device)
        bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
        temp = (im_flat + bias).mm(mat)
        out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
        # out = out.clamp(0, 1.0)
        return out


def main():
    vi_path = '/root/dataset/imagefusion/LLVIP_tiny_New/vi'
    fused_path = '/root/PycharmProjects/diffusion_for_mmif/output/LLVIP_Fusion_RGB_005I/recon/'
    save_path = '/root/PycharmProjects/diffusion_for_mmif/output/LLVIP_Fusion_RGB_005I/recon_color/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i, img_name in enumerate(os.listdir(vi_path)):
        print("%4d - %4d"%(i, len(vi_path)), end='\r')

        vi_img_path = os.path.join(vi_path, img_name)
        fused_img_path = os.path.join(fused_path, img_name)
        save_img_path = os.path.join(save_path, img_name)
        color_img(vi_img_path, fused_img_path, save_img_path)


def color_img(img_path, fused_img_path, save_img_path):

    im = cv2.imread(img_path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
    im_Y, im_cb, im_cr = im[..., 0], im[..., 1], im[..., 2]
    fused_img = cv2.imread(fused_img_path)
    fused_img = cv2.resize(fused_img, [im.shape[1], im.shape[0]])

    # import pdb;pdb.set_trace()
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2YCrCb)
    fused_img_Y, fused_img_cb, fused_img_cr = fused_img[..., 0], fused_img[..., 1], fused_img[..., 2]

    new_fused_img = np.stack([fused_img_Y, im_cb, im_cr], -1)
    new_fused_img = cv2.cvtColor(new_fused_img, cv2.COLOR_YCrCb2BGR)

    new_fused_img = ((new_fused_img)).astype(np.uint8)
    cv2.imwrite(save_img_path, new_fused_img)

if __name__ == '__main__':
    main()