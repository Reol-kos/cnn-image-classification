import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCIFAR10Model(nn.Module):
    def __init__(self):
        super(EnhancedCIFAR10Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 第一层卷积，通道数增加到 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 第二层卷积
            nn.ReLU(),
            nn.MaxPool2d(2),  # 最大池化层

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 第三层卷积，提取更高层次特征
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 第四层卷积
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),

            nn.Flatten(),  # 展平用于全连接层
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(512 * 4 * 4, 1024),  # 第一个全连接层
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),  # 第二个全连接层
            nn.ReLU(),
            nn.Linear(256, 10)  # 输出层，10 个分类
        )

    def forward(self, x):
        return self.model(x)
