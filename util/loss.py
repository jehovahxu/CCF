import torch
from torch import nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import time


# class DWA(AbsWeighting):
#     r"""Dynamic Weight Average (DWA).

#     This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
#     and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.

#     Args:
#         T (float, default=2.0): The softmax temperature.

#     """
#     def __init__(self):
#         super(DWA, self).__init__()

#     def backward(self, losses, **kwargs):
#         T = kwargs['T']
#         if self.epoch > 1:
#             w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(self.device)
#             batch_weight = self.task_num*F.softmax(w_i/T, dim=-1)
#         else:
#             batch_weight = torch.ones_like(losses).to(self.device)
#         loss = torch.mul(losses, batch_weight).sum()
#         loss.backward()
#         return batch_weight.detach().cpu().numpy()


class GradNormLoss(nn.Module):
    # 初始化函数
    def __init__(self, devices, num_of_task=3, alpha=1.5, topk=3):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task  # 设置任务数
        self.alpha = alpha  # 设置alpha值，用于控制梯度规范化的权重
        self.device = devices
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float, device=self.device))  # 初始化任务权重参数
        self.l1_loss = nn.L1Loss()
        self.L_0 = None  # 初始化基准损失
        self.topk = topk

    # 标准前向传播函数
    def forward(self, L_t: torch.Tensor):
        # 初始化初始损失 L_i_0
        if self.L_0 is None:
            self.L_0 = L_t.detach()  # 首次调用时保留输入的初始损失值，用detach防止其参与梯度计
        # 计算加权损失 w_i(t) * L_i(t)
        self.L_t = L_t
        # 只计算tok w的loss
        tmp_w = self.w
        # tmp_w[[0,1,2]] = [100, 100, 100]
        # tmp_w[0].data = torch.tensor(100)
        # tmp_w[1].data = torch.tensor(100)
        # tmp_w[2].data = torch.tensor(100)

        # top_k_values, top_k_indices = torch.topk(tmp_w, self.topk)
        top_k_values, top_k_indices = torch.topk(self.w, self.topk)

        # import pdb;pdb.set_trace()
        # print(top_k_indices)
        # self.wL_t = L_t[top_k_indices] * self.w[top_k_indices]
        self.wL_t = L_t[top_k_indices] * self.w[top_k_indices] + L_t[1]
        # self.wL_t = L_t * self.w  # 每个任务损失乘以对应的权重
        # 计算所有任务加权损失的总和
        # import pdb;pdb.set_trace()
        self.cv_loss = self.cv_squared(self.w)
        # self.total_loss = self.wL_t.sum() + self.cv_loss * 0.
        # self.total_loss = self.wL_t.sum() + self.cv_loss * 0.01
        self.total_loss = self.wL_t.sum()
        # 增加cv loss

        return self.total_loss, self.L_t, self.w, self.cv_loss, top_k_indices

    def cv_regularizer(self, gate_outputs):
        # 计算每个批次中每个专家的平均概率
        mean_usage = torch.mean(gate_outputs, axis=0)
        # 计算标准差
        std_deviation = torch.std(gate_outputs, axis=0)
        # 计算CV
        cv = std_deviation / (mean_usage + 1e-6)  # 加上一个小常数防止除以零
        # 计算CV的平均值作为正则项
        cv_loss = torch.mean(cv)
        return cv_loss

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        return x.float().var() / (x.float().mean() ** 2 + eps)

    # 额外的前向和反向传播过程
    def additional_forward_and_backward(self, grad_norm_weights, optimizer: torch.optim.Optimizer):
        # 在外部调用 optimizer.zero_grad()
        time0 = time.time()

        self.total_loss.backward(retain_graph=True)  # 对总损失进行反向传播，保留计算图
        time1 = time.time()
        # 在标准反向传播中，w的梯度不参与计算
        self.w.grad.data = self.w.grad.data * 0.0  # 清除权重w的梯度

        # import pdb;pdb.set_trace()
        self.GW_t = []
        for i in range(self.num_of_task):
            # 计算每个任务损失相对于共享参数的梯度
            GiW_t = torch.autograd.grad(self.L_t[i], grad_norm_weights, retain_graph=True)  # retain_graph保留计算图
            # 计算梯度的范数
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        time2 = time.time()
        self.GW_t = torch.stack(self.GW_t)  # 将所有任务的梯度范数堆叠起来
        self.bar_GW_t = self.GW_t.detach().mean()  # 计算梯度范数的平均值，并从计算图中分离
        self.tilde_L_t = (self.L_t / self.L_0).detach()  # 计算相对损失并分离计算图
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()  # 计算相对损失的比例
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))  # 计算梯度损失
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]  # 计算w的梯度
        lr = optimizer.state_dict()['param_groups'][0]['lr']  # 获取优化器的学习率
        optimizer.step()  # 执行优化步骤
        self.w.data = self.w.data - self.w.grad  # 更新权重w

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None  # 清理变量
        # 重新规范化w
        # import pdb;pdb.set_trace()
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task
        time3 = time.time()
        # 输出每个步骤的计算时间
        return self.w.data


class GradNormLoss_custom(nn.Module):
    # 初始化函数
    def __init__(self, devices, num_of_task=3, alpha=1.5):
        super(GradNormLoss_custom, self).__init__()
        self.num_of_task = num_of_task  # 设置任务数
        self.alpha = alpha  # 设置alpha值，用于控制梯度规范化的权重
        self.device = devices
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float, device=self.device))  # 初始化任务权重参数
        self.l1_loss = nn.L1Loss()
        self.L_0 = None  # 初始化基准损失

    # 标准前向传播函数
    def forward(self, L_t):
        # 初始化初始损失 L_i_0
        if self.L_0 is None:
            self.L_0 = L_t.detach()  # 首次调用时保留输入的初始损失值，用detach防止其参与梯度计
        # 计算加权损失 w_i(t) * L_i(t)
        self.L_t = L_t
        self.wL_t = L_t * self.w  # 每个任务损失乘以对应的权重
        # 计算所有任务加权损失的总和
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    def entropy_regularizer(self, gate_outputs):
        epsilon = 1e-6  # 避免对数为零的情况
        entropy = -torch.sum(gate_outputs * torch.log(gate_outputs + epsilon), axis=1)
        return torch.mean(entropy)

    def cv_regularizer(self, gate_outputs):
        # 计算每个批次中每个专家的平均概率
        mean_usage = torch.mean(gate_outputs, axis=0)
        # 计算标准差
        std_deviation = torch.std_mean(gate_outputs, axis=0)
        # 计算CV
        cv = std_deviation / (mean_usage + 1e-6)  # 加上一个小常数防止除以零
        # 计算CV的平均值作为正则项
        cv_loss = torch.mean(cv)
        return cv_loss

    # 额外的前向和反向传播过程
    def additional_forward_and_backward(self, grad_norm_weights, optimizer):
        # 在外部调用 optimizer.zero_grad()
        self.total_loss.backward(retain_graph=True)  # 对总损失进行反向传播，保留计算图
        # 在标准反向传播中，w的梯度不参与计算
        self.w.grad.data = self.w.grad.data * 0.0  # 清除权重w的梯度

        self.GW_t = []
        for i in range(self.num_of_task):
            # 计算每个任务损失相对于共享参数的梯度
            # GiW_t = torch.autograd.grad(self.L_t[i], grad_norm_weights, retain_graph=True)  # retain_graph保留计算图
            # 计算梯度的范数
            GiW_t = grad_norm_weights[i]
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        self.GW_t = torch.stack(self.GW_t)  # 将所有任务的梯度范数堆叠起来
        self.bar_GW_t = self.GW_t.detach().mean()  # 计算梯度范数的平均值，并从计算图中分离
        self.tilde_L_t = (self.L_t / self.L_0).detach()  # 计算相对损失并分离计算图
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()  # 计算相对损失的比例
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))  # 计算梯度损失
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]  # 计算w的梯度
        # lr = optimizer.state_dict()['param_groups'][0]['lr']  # 获取优化器的学习率
        # optimizer.step()  # 执行优化步骤
        self.w.data = self.w.data - self.w.grad  # 更新权重w

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None  # 清理变量
        # 重新规范化w
        # self.w.data = self.w.data / self.w.data.sum() * self.num_of_task

        self.w.data = torch.nn.functional.softmax(self.w.data) * self.num_of_task
        # import pdb;pdb.set_trace()
        return self.w.data

        # 输出每个步骤的计算时间
        # print


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.7
        self.beta = 1 - self.alpha

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp = (y * x).sum()
        fp = ((1 - y) * x).sum()
        fn = (y * (1 - x)).sum()
        # tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return 1 - tversky


def SoftIoULoss(pred, target):
    # Old One
    pred = torch.sigmoid(pred)
    smooth = 1

    # print("pred.shape: ", pred.shape)
    # print("target.shape: ", target.shape)

    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)

    loss = 1 - loss.mean()
    # loss = (1 - loss).mean()

    return loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return -_ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# def SSIM(img1,img2):
#     	# pdb.set_trace()
# 	img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
# 	img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0
# 	img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
# 	img2 = Variable( img2, requires_grad = False)
# 	# ssim_value = pytorch_ssim.ssim(img1, img2).item()
# 	ssim_value = float(ssim(img1, img2))
# 	# print(ssim_value)
# 	return ssim_value


def get_high_frequency_kernel(choose_kernel):
    kernelX = [[[-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0]]]
    kernelY = [[[-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0]]]  # out_channel,channels
    kernelX = torch.cuda.FloatTensor(kernelX).expand(1, 1, 3, 3)
    kernelY = torch.cuda.FloatTensor(kernelY).expand(1, 1, 3, 3)

    # print("model high frequency kernel is",choose_kernel)

    return kernelX, kernelY


class SobelLoss(nn.Module):
    def __init__(self, h_kernel=''):
        super(SobelLoss, self).__init__()
        self.kernelX, self.kernelY = get_high_frequency_kernel(h_kernel)
        self.weightX = nn.Parameter(data=self.kernelX, requires_grad=False)
        self.weightY = nn.Parameter(data=self.kernelY, requires_grad=False)

    def forward(self, pred, gt):
        fakeX = F.conv2d(pred, self.weightX, bias=None, stride=1, padding=1)
        fakeY = F.conv2d(pred, self.weightY, bias=None, stride=1, padding=1)
        # fake_sobel = torch.sqrt((fakeX*fakeX) + (fakeY*fakeY))
        fake_sobel = (torch.abs(fakeX) + torch.abs(fakeY)) / 2
        gtX = F.conv2d(gt, self.weightX, bias=None, stride=1, padding=1)
        gtY = F.conv2d(gt, self.weightY, bias=None, stride=1, padding=1)
        # gt_sobel = torch.sqrt((gtX*gtX) + (gtY*gtY))
        gt_sobel = (torch.abs(gtX) + torch.abs(gtY)) / 2
        # print(fake_sobel.shape, gt_sobel.shape)

        high_frequency_loss = F.l1_loss(fake_sobel, gt_sobel)

        return high_frequency_loss  # ,gt_sobel,fake_sobel

# class AbsWeighting(nn.Module):
#     r"""An abstract class for weighting strategies.
#     """
#     def __init__(self):
#         super(AbsWeighting, self).__init__()

#     def init_param(self):
#         r"""Define and initialize some trainable parameters required by specific weighting methods.
#         """
#         pass

#     def _compute_grad_dim(self):
#         self.grad_index = []
#         for param in self.get_share_params():
#             self.grad_index.append(param.data.numel())
#         self.grad_dim = sum(self.grad_index)

#     def _grad2vec(self):
#         grad = torch.zeros(self.grad_dim)
#         count = 0
#         for param in self.get_share_params():
#             if param.grad is not None:
#                 beg = 0 if count == 0 else sum(self.grad_index[:count])
#                 end = sum(self.grad_index[:(count+1)])
#                 grad[beg:end] = param.grad.data.view(-1)
#             count += 1
#         return grad

#     def _compute_grad(self, losses, mode, rep_grad=False):
#         '''
#         mode: backward, autograd
#         '''
#         if not rep_grad:
#             grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
#             for tn in range(self.task_num):
#                 if mode == 'backward':
#                     losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
#                     grads[tn] = self._grad2vec()
#                 elif mode == 'autograd':
#                     grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
#                     grads[tn] = torch.cat([g.view(-1) for g in grad])
#                 else:
#                     raise ValueError('No support {} mode for gradient computation')
#                 self.zero_grad_share_params()
#         else:
#             if not isinstance(self.rep, dict):
#                 grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
#             else:
#                 grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
#             for tn, task in enumerate(self.task_name):
#                 if mode == 'backward':
#                     losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
#                     grads[tn] = self.rep_tasks[task].grad.data.clone()
#         return grads

#     def _reset_grad(self, new_grads):
#         count = 0
#         for param in self.get_share_params():
#             if param.grad is not None:
#                 beg = 0 if count == 0 else sum(self.grad_index[:count])
#                 end = sum(self.grad_index[:(count+1)])
#                 param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
#             count += 1

#     def _get_grads(self, losses, mode='backward'):
#         r"""This function is used to return the gradients of representations or shared parameters.

#         If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
#         the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
#         The second element is the resized gradients with size of [task_num, -1], which means \
#         the gradient of each task is resized as a vector.

#         If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
#         of [task_num, -1], which means the gradient of each task is resized as a vector.
#         """
#         if self.rep_grad:
#             per_grads = self._compute_grad(losses, mode, rep_grad=True)
#             if not isinstance(self.rep, dict):
#                 grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
#             else:
#                 try:
#                     grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
#                 except:
#                     raise ValueError('The representation dimensions of different tasks must be consistent')
#             return [per_grads, grads]
#         else:
#             self._compute_grad_dim()
#             grads = self._compute_grad(losses, mode)
#             return grads

#     def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
#         r"""This function is used to reset the gradients and make a backward.

#         Args:
#             batch_weight (torch.Tensor): A tensor with size of [task_num].
#             per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
#             grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters.
#         """
#         if self.rep_grad:
#             if not isinstance(self.rep, dict):
#                 # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
#                 transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
#                 self.rep.backward(transformed_grad)
#             else:
#                 for tn, task in enumerate(self.task_name):
#                     rg = True if (tn+1)!=self.task_num else False
#                     self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
#         else:
#             # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
#             new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
#             self._reset_grad(new_grads)

#     @property
#     def backward(self, losses, **kwargs):
#         r"""
#         Args:
#             losses (list): A list of losses of each task.
#             kwargs (dict): A dictionary of hyperparameters of weighting methods.
#         """
#         pass
