import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

device = torch.device("cpu")


def ssim_loss_vi(body_out_vi1, vicut):
    """可见光图像的结构相似性损失"""
    ssim_loss = ssim(body_out_vi1, vicut)
    return ssim_loss


def ssim_loss_ir(body_out_ir1, ircut):
    """红外图像的结构相似性损失"""
    ssim_loss_ir = ssim(body_out_ir1, ircut)
    return ssim_loss_ir


def ssim_loss_vi1(body_out_vi1, vicut):
    """可见光图像的结构相似性损失（副本）"""
    ssim_loss_vi1 = ssim(body_out_vi1, vicut)
    return ssim_loss_vi1


def ssim_loss_ir1(body_out_ir1, ircut):
    """红外图像的结构相似性损失（副本）"""
    ssim_loss_ir1 = ssim(body_out_ir1, ircut)
    return ssim_loss_ir1


def ssim_loss_vilow(imageoutput, vicutlow):
    """低分辨率可见光图像的结构相似性损失"""
    ssim_loss = ssim(imageoutput, vicutlow)
    return ssim_loss


def ssim_loss_irlow(imageoutput, ircutlow):
    """低分辨率红外图像的结构相似性损失"""
    ssim_loss_ir = ssim(imageoutput, ircutlow)
    return ssim_loss_ir


def gaussian(window_size, sigma):
    """生成高斯核"""
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    """创建高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, val_range=None):
    """计算结构相似性指数 (SSIM)"""
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0

        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # 计算均值
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    # 计算方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # SSIM常数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算SSIM
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    return 1 - ssim_map.mean()




def loss_sobel1(input):
    """使用Sobel算子计算图像梯度"""
    # 定义Sobel卷积核
    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1,
                        bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1,
                        bias=False, padding=1, stride=1)

    # 设置水平方向Sobel算子
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()

    # 设置垂直方向Sobel算子
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    # 计算梯度
    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)

    return image_gradient