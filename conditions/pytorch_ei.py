import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp



def EI(cls, img):  # Average gradient
    ### different sobel kernel has different results
    ei = 0.
    for i in range(img.shape[0]):
        edgesh = filters.sobel_h(img[i])
        edgesv = filters.sobel_v(img[i])
        ei += np.mean(np.sqrt(edgesh ** 2 + edgesv ** 2))
    return ei / img.shape[0]