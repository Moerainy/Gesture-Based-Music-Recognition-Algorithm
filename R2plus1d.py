"""
仅用作模型理解，不要运行。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalConv(nn.Module):
    """
    空间-时间分解卷积模块：将 3D 卷积分解为 2D 空间卷积 + 1D 时间卷积
    根据论文 "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, first_conv=False):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小 (temporal_size, height_size, width_size)
            stride: 步长
            padding: 填充
            bias: 是否使用偏置
            first_conv: 是否为网络的第一层卷积（有特殊的核大小和中间通道数）
        """
        super(SpatioTemporalConv, self).__init__()
        
        # 分解卷积核大小
        if isinstance(kernel_size, tuple):
            temporal_kernel_size, spatial_kernel_size, _ = kernel_size
        else:
            temporal_kernel_size = spatial_kernel_size = kernel_size
        
        # 计算中间通道数（连接空间和时间卷积的桥梁）
        # 公式: M = floor((t * h * w * in_c * out_c) / (h * w * in_c + t * out_c))
        if first_conv:
            # 第一层使用固定的中间通道数 45（来自 torchvision 实现）
            self.intermediate_channels = 45
        else:
            # 对于普通层，使用公式计算最优中间通道数
            t = temporal_kernel_size
            h = spatial_kernel_size
            w = spatial_kernel_size
            numerator = t * h * w * in_channels * out_channels
            denominator = h * w * in_channels + t * out_channels
            self.intermediate_channels = numerator // denominator
        
        # 空间卷积（2D）：处理 H x W 维度
        self.spatial_conv = nn.Conv2d(
            in_channels, self.intermediate_channels,
            kernel_size=(spatial_kernel_size, spatial_kernel_size),
            stride=(stride if isinstance(stride, tuple) else (stride, stride))[1:],
            padding=(padding if isinstance(padding, tuple) else (padding, padding))[1:],
            bias=bias
        )
        
        # 时间卷积（1D）：处理 T 维度
        self.temporal_conv = nn.Conv3d(
            self.intermediate_channels, out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=(stride if isinstance(stride, tuple) else (stride, 1, 1)),
            padding=(padding if isinstance(padding, tuple) else (padding, 0, 0)),
            bias=bias
        )
        
        # 批归一化层
        self.bn_spatial = nn.BatchNorm2d(self.intermediate_channels)
        self.bn_temporal = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch, channels, time, height, width)
        Returns:
            输出张量，形状为 (batch, out_channels, time, height, width)
        """
        # 获取输入形状
        batch, channels, time, height, width = x.shape
        
        # 空间卷积：合并 batch 和 time 维度
        # 将 (batch, channels, time, height, width) -> (batch*time, channels, height, width)
        x_spatial = x.permute(0, 2, 1, 3, 4).contiguous()
        x_spatial = x_spatial.view(batch * time, channels, height, width)
        
        # 应用空间卷积
        out = self.spatial_conv(x_spatial)
        out = self.bn_spatial(out)
        out = self.relu(out)
        
        # 恢复时间维度
        _, inter_channels, h_out, w_out = out.shape
        out = out.view(batch, time, inter_channels, h_out, w_out)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        
        # 时间卷积
        out = self.temporal_conv(out)
        out = self.bn_temporal(out)
        
        return out


class BasicBlock3D(nn.Module):
    """R(2+1)D 的基本残差块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        
        # 第一个分解卷积层
        self.conv1 = SpatioTemporalConv(
            in_channels, out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=(1, 1, 1),
            bias=False,
            first_conv=False
        )
        
        # 第二个分解卷积层（stride=1）
        self.conv2 = SpatioTemporalConv(
            out_channels, out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
            bias=False,
            first_conv=False
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 残差连接，处理维度不匹配的情况
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class VideoResNet(nn.Module):
    """视频 ResNet 基类，支持 R(2+1)D 结构"""
    
    def __init__(self, block, layers, num_classes=400, input_channels=3):
        """
        Args:
            block: 残差块类型
            layers: 每层的块数量列表，如 [2, 2, 2, 2] 对应 18 层
            num_classes: 分类数（Kinetics-400 默认 400 类）
            input_channels: 输入通道数
        """
        super(VideoResNet, self).__init__()
        
        self.in_channels = 64
        
        # 第一层：特殊的空间卷积（只在空间维度下采样）
        self.conv1 = SpatioTemporalConv(
            input_channels, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
            first_conv=True
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 最大池化（只在空间维度下采样）
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # 4 个残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分类器
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建包含多个残差块的层"""
        downsample = None
        
        # 如果维度不匹配，创建下采样层
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                SpatioTemporalConv(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=(1, 1, 1),
                    stride=stride,
                    bias=False,
                    first_conv=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入视频张量
               形状: (batch, channels, time, height, width)
        Returns:
            输出 logits，形状: (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def r2plus1d_18(num_classes=400, pretrained=False, **kwargs):
    """
    构建 R(2+1)D-18 网络
    
    Args:
        num_classes: 分类数（默认 400，对应 Kinetics-400）
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数
    
    Returns:
        VideoResNet 模型
    """
    model = VideoResNet(
        BasicBlock3D,
        [2, 2, 2, 2],  # ResNet18 的层配置
        num_classes=num_classes,
        **kwargs
    )
    
    # 注意：预训练权重的加载需要额外的处理
    # 这里只提供模型结构，预训练权重可从 torchvision 官方下载
    if pretrained:
        print("Warning: Pre-trained weights not included in this implementation.")
        print("Please use torchvision.models.video.r2plus1d_18(pretrained=True) for official version.")
    
    return model


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = r2plus1d_18(num_classes=400)
    
    # 打印模型信息
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(model)
    
    # # 测试前向传播
    # # 输入形状：(batch, channels, time, height, width)
    # # 标准 Kinetics-400 输入：8帧，112x112 空间分辨率
    # x = torch.randn(1, 3, 16, 112, 112)
    # print(f"\nInput shape: {x.shape}")
    
    # with torch.no_grad():
    #     output = model(x)
    
    # print(f"Output shape: {output.shape}")
    
    # # 测试不同输入长度
    # x_short = torch.randn(2, 3, 8, 112, 112)
    # output_short = model(x_short)
    # print(f"\nBatch size 2, 8 frames: {output_short.shape}")