import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F


# 原始图像 (210, 160, 3)
img_transform = T.Compose([
   T.ToPILImage(),   # 将输入的 NumPy 数组或 Tensor 转为 PIL 图像
   T.Lambda(lambda img: F.crop(img, top=34, left=0, height=160, width=160)),  # 裁剪行 [34:194]
   T.Grayscale(num_output_channels=1), # 转为灰度图像
   T.ToTensor(),
   T.Lambda(lambda x: (x > 0.35).float()),   # 将非零像素值变为 1 (二值化)
   T.Resize((80, 80), interpolation=T.InterpolationMode.NEAREST), # 缩放图像到 80x80, 插值防模糊
   nn.Flatten(start_dim=1)
])