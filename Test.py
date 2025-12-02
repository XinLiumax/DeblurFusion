import os
import time

import torch
import numpy as np
import imageio
import torchvision.transforms as transforms

from PIL import Image
from Networks.model import DeblurFusion as Net


def load_model(model_path):
    """加载预训练模型"""
    model = Net()

    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

    model.eval()
    return model


def process_image_pair(model, ir_path, vi_path, output_dir):
    """处理一对红外和可见光图像"""
    with torch.no_grad():
        # 读取图像
        ir_img = Image.open(ir_path).convert('L')
        vi_img = Image.open(vi_path).convert('L')

        # 转换为Tensor
        transform = transforms.ToTensor()
        ir_tensor = transform(ir_img).unsqueeze(0)
        vi_tensor = transform(vi_img).unsqueeze(0)

        # 移动到GPU
        if torch.cuda.is_available():
            ir_tensor = ir_tensor.cuda()
            vi_tensor = vi_tensor.cuda()

        # 模型推理
        fused, _, _ = model(ir_tensor, vi_tensor)

        # 转换为numpy数组并保存
        result = (np.squeeze(fused.cpu().numpy()) * 255).astype(np.uint8)
        filename = os.path.basename(ir_path)
        output_path = os.path.join(output_dir, filename)

        imageio.imwrite(output_path, result)


def batch_process(ir_dir, vi_dir, output_dir, model_path):
    """批量处理图像对"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载模型
    model = load_model(model_path)

    # 获取图像文件列表
    image_extensions = ('.bmp', '.png', '.jpg')
    ir_files = sorted([f for f in os.listdir(ir_dir)
                       if f.endswith(image_extensions)])
    vi_files = sorted([f for f in os.listdir(vi_dir)
                       if f.endswith(image_extensions)])

    # 批量处理
    start_time = time.time()

    for ir_file, vi_file in zip(ir_files, vi_files):
        ir_path = os.path.join(ir_dir, ir_file)
        vi_path = os.path.join(vi_dir, vi_file)

        process_image_pair(model, ir_path, vi_path, output_dir)

    # 打印处理时间
    elapsed_time = time.time() - start_time
    print(f'处理完成，共处理 {len(ir_files)} 对图片，'
          f'耗时: {elapsed_time:.2f}秒')


if __name__ == '__main__':
    # 路径配置
    IR_FOLDER = r''
    VI_FOLDER = r''
    OUTPUT_FOLDER = './results'
    MODEL_PATH = './model.pth'

    # 执行批量处理
    batch_process(IR_FOLDER, VI_FOLDER, OUTPUT_FOLDER, MODEL_PATH)