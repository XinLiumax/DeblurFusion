import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import pandas as pd
import joblib
import glob
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from Networks.model import DeblurFusion as net
from losses import (
    ssim_loss_ir, ssim_loss_vi, ssim_loss_ir1, ssim_loss_vi1,
    ssim_loss_irlow, ssim_loss_vilow, loss_sobel1
)

device = torch.device('cuda')
use_gpu = torch.cuda.is_available()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model_name',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float)
    parser.add_argument('--weight', default=[1, 1, 1, 1, 1, 1, 10],
                        type=float, nargs='+')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    # 使用空列表作为默认参数以支持直接运行
    args = parser.parse_args(args=[])

    return args


class GetDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, transform=None):
    
        self.transform = transform

        # 路径定义
        self.ir_dir = r'./'
        self.vi_dir = r'./'
        self.ir_high_dir = r'./'
        self.vi_high_dir = r'./'

        self.ir_list = natsorted(os.listdir(self.ir_dir))

    def __getitem__(self, index):
        image_name = self.ir_list[index]

        # 构建完整路径
        ircutlow_path = os.path.join(self.ir_dir, image_name)
        vicutlow_path = os.path.join(self.vi_dir, image_name)
        ircut_path = os.path.join(self.ir_high_dir, image_name)
        vicut_path = os.path.join(self.vi_high_dir, image_name)

        # 读取图像
        ircutlow = Image.open(ircutlow_path).convert('L')
        vicutlow = Image.open(vicutlow_path).convert('L')
        ircut = Image.open(ircut_path).convert('L')
        vicut = Image.open(vicut_path).convert('L')

        # 转换为Tensor
        if self.transform is not None:
            tran = transforms.ToTensor()

            ircutlow = tran(ircutlow)
            vicutlow = tran(vicutlow)
            ircut = tran(ircut)
            vicut = tran(vicut)

            input_tensor = torch.cat((ircutlow, vicutlow), -3)

            return input_tensor, ircutlow, vicutlow, ircut, vicut

    def __len__(self):
        return len(self.ir_list)


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader_ir, train_loader_vi, model,
          criterion_ssim_ir, criterion_ssim_vi, criterion_ssim_ir1,
          criterion_ssim_vi1, criterion_ssim_irlow, criterion_ssim_vilow,
          criterion_sobel, optimizer, epoch):
    """训练一个epoch"""
    # 初始化记录器
    losses = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    losses_ssim_ir1 = AverageMeter()
    losses_ssim_vi1 = AverageMeter()
    losses_ssim_irlow = AverageMeter()
    losses_ssim_vilow = AverageMeter()
    losses_sobel = AverageMeter()

    weight = args.weight
    model.train()

    for i, (input_tensor, ircutlow, vicutlow, ircut, vicut) in \
            tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):
        # 数据移动到GPU
        input_tensor = input_tensor.cuda()
        ircutlow = ircutlow.cuda()
        vicutlow = vicutlow.cuda()
        ircut = ircut.cuda()
        vicut = vicut.cuda()

        # 前向传播
        outputs = model(ircutlow, vicutlow)
        imageoutput, body_out_ir3, body_out_vi3 = outputs

        # 计算各项损失
        loss_ssim_ir = weight[0] * criterion_ssim_ir(imageoutput, ircut)
        loss_ssim_vi = weight[1] * criterion_ssim_vi(imageoutput, vicut)
        loss_ssim_irlow = weight[2] * criterion_ssim_irlow(imageoutput, ircutlow)
        loss_ssim_vilow = weight[3] * criterion_ssim_vilow(imageoutput, vicutlow)
        loss_ssim_ir1 = weight[4] * criterion_ssim_ir1(body_out_ir3, ircut)
        loss_ssim_vi1 = weight[5] * criterion_ssim_vi1(body_out_vi3, vicut)

        loss_smooth = nn.SmoothL1Loss(reduction='mean')
        loss_sobel_val = weight[6] * loss_smooth(
            criterion_sobel(imageoutput),
            torch.max(criterion_sobel(ircut), criterion_sobel(vicut))
        )

        # 总损失
        total_loss = (loss_ssim_ir + loss_ssim_vi + loss_ssim_ir1 +
                      loss_ssim_vi1 + loss_ssim_irlow + loss_ssim_vilow +
                      loss_sobel_val)

        # 更新记录器
        batch_size = ircutlow.size(0)
        losses.update(total_loss.item(), batch_size)
        losses_ssim_ir.update(loss_ssim_ir.item(), batch_size)
        losses_ssim_vi.update(loss_ssim_vi.item(), batch_size)
        losses_ssim_irlow.update(loss_ssim_irlow.item(), batch_size)
        losses_ssim_vilow.update(loss_ssim_vilow.item(), batch_size)
        losses_sobel.update(loss_sobel_val.item(), batch_size)
        losses_ssim_ir1.update(loss_ssim_ir1.item(), batch_size)
        losses_ssim_vi1.update(loss_ssim_vi1.item(), batch_size)

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 返回日志记录
    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ssim_ir', losses_ssim_ir.avg),
        ('loss_ssim_vi', losses_ssim_vi.avg),
        ('loss_ssim_irlow', losses_ssim_irlow.avg),
        ('loss_ssim_vilow', losses_ssim_vilow.avg),
        ('loss_sobel', losses_sobel.avg),
        ('loss_ssim_ir1', losses_ssim_ir1.avg),
        ('loss_ssim_vi1', losses_ssim_vi1.avg),
    ])

    return log


def main():
    """主训练函数"""
    args = parse_args()

    # 创建模型保存目录
    model_dir = f'models/{args.name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 打印和保存参数
    print('配置参数 -----')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('------------')

    with open(f'{model_dir}/args.txt', 'w') as f:
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}', file=f)

    joblib.dump(args, f'{model_dir}/args.pkl')
    cudnn.benchmark = True


    # 数据转换
    transform_trainir = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.455, 0.455, 0.455), (0.215, 0.215, 0.215))
    ])

    transform_trainvi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.422, 0.424, 0.413), (0.243, 0.240, 0.256))
    ])

    # 创建数据集和数据加载器
    dataset_train_ir = GetDataset(transform=transform_trainir)
    dataset_train_vi = GetDataset(transform=transform_trainvi)

    train_loader_ir = DataLoader(
        dataset_train_ir,
        shuffle=True,
        batch_size=args.batch_size
    )
    train_loader_vi = DataLoader(
        dataset_train_vi,
        shuffle=True,
        batch_size=args.batch_size
    )

    # 初始化模型
    model = net()
    model = model.cuda()

    # 损失函数
    criterion_ssim_ir = ssim_loss_ir
    criterion_ssim_vi = ssim_loss_vi
    criterion_ssim_irlow = ssim_loss_irlow
    criterion_ssim_vilow = ssim_loss_vilow
    criterion_sobel = loss_sobel1
    criterion_ssim_ir1 = ssim_loss_ir1
    criterion_ssim_vi1 = ssim_loss_vi1

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps
    )

    # 初始化日志记录
    log = pd.DataFrame(
        columns=[
            'epoch', 'loss', 'loss_ssim_ir', 'loss_ssim_vi',
            'loss_ssim_ir1', 'loss_ssim_vi1', 'loss_ssim_irlow',
            'loss_ssim_vilow', 'loss_sobel'
        ]
    )

    # 训练循环
    for epoch in range(args.epochs):
        print(f'Epoch [{epoch + 1}/{args.epochs}]')

        train_log = train(
            args, train_loader_ir, train_loader_vi, model,
            criterion_ssim_ir, criterion_ssim_vi, criterion_ssim_ir1,
            criterion_ssim_vi1, criterion_ssim_irlow, criterion_ssim_vilow,
            criterion_sobel, optimizer, epoch
        )

        # 打印训练结果
        print(
            'loss: %.4f - loss_ssim_ir: %.4f - loss_ssim_vi: %.4f - '
            'loss_ssim_ir1: %.4f - loss_ssim_vi1: %.4f - '
            'loss_ssim_irlow: %.4f - loss_ssim_vilow: %.4f - '
            'loss_sobel: %.4f' % (
                train_log['loss'],
                train_log['loss_ssim_ir'],
                train_log['loss_ssim_vi'],
                train_log['loss_ssim_ir1'],
                train_log['loss_ssim_vi1'],
                train_log['loss_ssim_irlow'],
                train_log['loss_ssim_vilow'],
                train_log['loss_sobel']
            )
        )

        # 记录日志
        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['loss_ssim_ir'],
            train_log['loss_ssim_vi'],
            train_log['loss_ssim_ir1'],
            train_log['loss_ssim_vi1'],
            train_log['loss_ssim_irlow'],
            train_log['loss_ssim_vilow'],
            train_log['loss_sobel']
        ], index=[
            'epoch', 'loss', 'loss_ssim_ir', 'loss_ssim_vi',
            'loss_ssim_ir1', 'loss_ssim_vi1', 'loss_ssim_irlow',
            'loss_ssim_vilow', 'loss_sobel'
        ])

        log = log._append(tmp, ignore_index=True)
        log.to_csv(f'{model_dir}/log.csv', index=False)

        # 保存模型
        if (epoch + 1) % 1 == 0:
            torch.save(
                model.state_dict(),
                f'./weights/model_{epoch + 1}.pth'
            )

        # 学习率调整
        if epoch + 1 == 25:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1


if __name__ == '__main__':
    main()