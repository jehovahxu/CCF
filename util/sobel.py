import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from PIL import Image
from torch.autograd import Variable


def nn_conv2d(im):
    conv_op = nn.Conv2d(1,1,3,bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1,1,3,3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    edge_detect = conv_op(Variable(im))
    edge_detect = edge_detect.squeeze().detech().numpy()

    return edge_detect

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
        # sobel_kernel = sobel_kernel.repeat(3, axis=1)
        self.weight = Variable(torch.from_numpy(sobel_kernel)).to(devices)
    def sobel(self, im):
        if im.shape[1] == 3:
            im = rgb2gray(im)
        im = (im + 1) * 127.5
        edge_detect = F.conv2d(Variable(im), self.weight, padding=1)
        edge_detect = torch.clip(edge_detect, 0, 255)
        edge_detect = edge_detect/127.5 - 1
        return edge_detect

def main():
    im = Image.open('/root/dataset/imagefusion/LLVIP_tiny_New/vi/190015.jpg').convert('L')
    im = np.array(im, dtype='float32')
    im = torch.from_numpy(im.reshape(((1, 1, im.shape[0], im.shape[1]))))
    im = im/127.5 - 1
    device_str = f"cuda:{0}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    sobel = Sobel(device)
    im = im.to(device)
    edge_detect = sobel.sobel(im)
    edge_detect = edge_detect.squeeze().detach().cpu().numpy()
    # edge_detect = (edge_detect + 1) * 255
    # edge_detect = (edge_detect - np.min(edge_detect)) / (np.max(edge_detect) - np.min(edge_detect))
    edge_detect = (edge_detect +1) * 127.5
    # edge_detect = (np.clip(edge_detect, -1, 1) + 1 )* 127.5
    # edge_detect = edge_detect[np.newaxis, ...].repeat(3, axis=0)
    # im = Image.fromarray(edge_detect.transpose((1,2,0)), mode='RGB')
    im = Image.fromarray(edge_detect)
    im = im.convert('L')
    im.save('./test.png', quality=95)


if __name__ == '__main__':
    main()