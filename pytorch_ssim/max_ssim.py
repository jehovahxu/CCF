
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import sys
sys.path.append('/root/PycharmProjects/diffusion_for_mmif/.')
sys.path.append('/root/PycharmProjects/diffusion_for_mmif/pytorch_ssim')
import pytorch_ssim

npImg1 = cv2.imread("/root/dataset/imagefusion/LLVIP_tiny_New/vi/260468.jpg")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/127.5 -1
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable(img1,  requires_grad=False)
img2 = Variable(img2, requires_grad=True)
# import pdb;pdb.set_trace()

# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).data
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)
i = 0
import pdb;pdb.set_trace()
while ssim_value < 0.95:
    i += 1
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.data
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()

print(i)