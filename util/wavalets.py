import torch
import os
import cv2
import numpy as np
import pywt
import torchvision
from PIL import Image
from torchvision import transforms as trans
from torchvision.utils import save_image
from pytorch_wavelets import DWTForward, DWTInverse
from glob import glob

class Wavalets():
    def __init__(self, devices=None, J=3):
        if devices is not None:
            self.xfm = DWTForward(J=J, mode='zero', wave='haar').to(devices)  # Accepts all wave types available to PyWavelets
            self.ifm = DWTInverse(mode='zero', wave='haar').to(devices)
        else:
            self.xfm = DWTForward(J=J, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
            self.ifm = DWTInverse(mode='zero', wave='haar')
        ### for multi-wavalets test
        self.multi_xfm  = [DWTForward(J=1, mode='zero', wave='db2').to(devices),
                           DWTForward(J=2, mode='zero', wave='db2').to(devices),
                           DWTForward(J=3, mode='zero', wave='db2').to(devices)
                           ]
        self.multi_ifm  = DWTInverse(mode='zero', wave='db2').to(devices)

    def forward(self, img):
        """
        Args:
            img: tensor

        Returns:

        """

        Yl, Yh = self.xfm(img)
        # Yl = trans.Resize((256, 256))(Yl)
        return Yl, Yh
    def inverse(self, Yl, Yh):
        coeffs = (Yl, Yh)
        img = self.ifm(coeffs)
        return img

    def wavalets_multi(self, visible, infrared, pred_xstart):

        for i in range(3):
            inf_yl, inf_yh = self.multi_xfm[i](infrared)
            vis_yl, vis_yh = self.multi_xfm[i](visible)
            x_start_yl, x_start_yh = self.multi_xfm[i](pred_xstart)
            new_yh = vis_yh
            new_yl = (inf_yl + vis_yl) / 2
            for i in range(len(new_yh)):
                new_yh[i] = torch.where(inf_yh[i] > vis_yh[i], inf_yh[i], vis_yh[i])
                x_start_yh[i] = x_start_yh[i] - new_yh[i]
            x_start_yl = x_start_yl - new_yl
            new_x_start = wavalets.inverse(x_start_yl, x_start_yh)

        # import pdb;pdb.set_trace()
        new_x_start = wavalets.inverse(x_start_yl, x_start_yh)
        return new_x_start

    def wavalets_test(self):
        transform = trans.Compose([
            # trans.Resize((256, 256)),
            trans.ToTensor()
        ])
        inf_img = Image.open('/root/dataset/imagefusion/M3FD_Fusion/ir/00000.png').convert('L')
        vis_img = Image.open('/root/dataset/imagefusion/M3FD_Fusion/vi/00000.png').convert('L')
        vis_img = transform(vis_img).unsqueeze(0)
        inf_img = transform(inf_img).unsqueeze(0)
        xfm = DWTForward(J=3, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets

        vis_img_yl, vis_img_yh = xfm(vis_img)
        inf_img_yl, inf_img_yh = xfm(inf_img)
        new_yl = (inf_img_yl + vis_img_yl) / 2
        new_yh = vis_img_yh
        for i in range(len(vis_img_yh)):
            new_yh[i] = torch.where(torch.abs(inf_img_yh[i]) > torch.abs(vis_img_yh[i]), inf_img_yh[i], vis_img_yh[i])
        ifm = DWTInverse(mode='zero', wave='haar')
        new_img = ifm((new_yl, new_yh))
        save_image(new_img,'./test.png')
        import pdb;pdb.set_trace()


        return new_img

    def wavalets_test_1(self):
        inf_path = '/root/dataset/imagefusion/LLVIP_tiny_New/ir/'
        vis_path = '/root/dataset/imagefusion/LLVIP_tiny_New/vi/'
        save_path = '/root/PycharmProjects/diffusion_for_mmif/output/M3FD_Fusion_wavalets_multi'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # file_list = glob(os.path.join(inf_path, '*.png'))
        for img_name in sorted(os.listdir(inf_path)):
            print(f'process {img_name}')
            inf_name = os.path.join(inf_path, img_name)
            vis_name = os.path.join(vis_path, img_name)
            transform = trans.Compose([
                # trans.Resize((256, 256)),
                trans.ToTensor()
            ])
            inf_img = Image.open(inf_name).convert('RGB')
            vis_img = Image.open(vis_name).convert('RGB')
            vis_img = transform(vis_img).unsqueeze(0)
            inf_img = transform(inf_img).unsqueeze(0)
            xfm = DWTForward(J=2, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets

            vis_img_yl, vis_img_yh = xfm(vis_img)
            inf_img_yl, inf_img_yh = xfm(inf_img)
            new_yl = (inf_img_yl + vis_img_yl) / 2
            new_yh = vis_img_yh
            # import pdb;pdb.set_trace()
            for i in range(len(vis_img_yh)):
                # new_yh[i] = torch.max(inf_img_yh[i], vis_img_yh[i])
                new_yh[i] = torch.where(torch.abs(inf_img_yh[i]) > torch.abs(vis_img_yh[i]), inf_img_yh[i],
                                        vis_img_yh[i])
                # new_yh[i][:, :, 0, ...] = torch.max(inf_img_yh[i][:, :, 0, ...], vis_img_yh[i][:, :, 0, ...])
                # new_yh[i][:, :, 1, ...] = torch.max(inf_img_yh[i][:, :, 1, ...], vis_img_yh[i][:, :, 1, ...])
                # new_yh[i][:, :, 2, ...] = torch.max(inf_img_yh[i][:, :, 2, ...], vis_img_yh[i][:, :, 2, ...])
            ifm = DWTInverse(mode='zero', wave='haar')
            new_img = ifm((new_yl, new_yh))
            save_image(new_img,os.path.join(save_path, img_name))


    def wavalets_test_2(self):
        def fuseCoeff(cooef1, cooef2, method):
            if (method == 'mean'):
                cooef = (cooef1 + cooef2) / 2
            elif (method == 'min'):
                cooef = np.minimum(cooef1, cooef2)
            elif (method == 'max'):
                cooef = np.maximum(cooef1, cooef2)
            return cooef

        # Params
        FUSION_METHOD = 'mean'  # Can be 'min' || 'max || anything you choose according theory
        FUSION_METHOD1 = 'max'
        # Read the two image
        I1 = cv2.imread('/root/dataset/imagefusion/LLVIP_tiny_New/ir/190015.jpg')
        I2 = cv2.imread('/root/dataset/imagefusion/LLVIP_tiny_New/vi/190015.jpg')
        # First: Do wavelet transform on each image
        wavelet = 'db2'
        cooef1 = pywt.wavedec2(I1[:, :], wavelet, level=3)
        cooef2 = pywt.wavedec2(I2[:, :], wavelet, level=3)
        # Second: for each level in both image do the fusion according to the desire option
        fusedCooef = []
        for i in range(len(cooef1)):
            # The first values in each decomposition is the apprximation values of the top level
            if (i == 0):
                fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0], FUSION_METHOD))
            else:
                # For the rest of the levels we have tupels with 3 coeeficents
                c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], FUSION_METHOD1)
                c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD1)
                c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD1)
                fusedCooef.append((c1, c2, c3))
        # Third: After we fused the cooefficent we nned to transfor back to get the image
        fusedImage = pywt.waverec2(fusedCooef, wavelet)
        # Forth: normmalize values to be in uint8
        fusedImage1 = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))),
                                  255)
        fusedImage1 = fusedImage1.astype(np.uint8)
        # Fith: Show image
        cv2.imwrite("./test.png", fusedImage1)


        return fusedImage1

if __name__ == '__main__':

    device_str = f"cuda:{0}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    wavalets = Wavalets(device)
    wavalets.wavalets_test_1()

