import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedConv2d(nn.Module):
    """PyTorch implementation of Gated Convolution layer.

    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of output filters.
        kernel_size (int or tuple): Size of the convolution kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding for the convolution.
        dilation (int or tuple): Dilation of the convolution.
        groups (int): Number of blocked connections.
        bias (bool): Whether to use bias.
        **kwargs: Other convolution arguments.
    """

    def __init__(self, in_channels: int, n_filters: int = 64,
                 kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1),
                 groups: int = 1, bias: bool = True, **kwargs):
        super(GatedConv2d, self).__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters

        # 标准卷积层，输出通道数是n_filters的两倍
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution weights."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, n_filters, height, width)
        """
        # 应用卷积，输出通道数是n_filters的两倍
        output = self.conv(x)  # (batch_size, n_filters*2, height, width)

        # 分割为线性部分和门控部分
        linear_part = output[:, :self.n_filters, :, :]  # 前n_filters个通道
        gate_part = output[:, self.n_filters:, :, :]  # 后n_filters个通道

        # 对门控部分应用sigmoid激活函数
        gate_signal = torch.sigmoid(gate_part)

        # 应用门控机制：线性部分 * 门控信号
        gated_output = linear_part * gate_signal

        return gated_output


def cram_block(x: torch.Tensor, n_filters: int = 64,
               pool_size: Tuple[int, int] = (2, 2),
               dropout_rate: float = 0.2) -> torch.Tensor:
    """Apply two gated convolutions followed by a max-pooling operation.

    Batch normalization and dropout are applied for regularization.

    Args:
        x (torch.Tensor): Input tensor to transform.
        n_filters (int): Number of filters for each gated convolution.
        pool_size (int or tuple): Pool size of max-pooling operation.
        dropout_rate (float): Fraction of units to drop.

    Returns:
        torch.Tensor: The resulting output tensor.
    """
    # 获取输入通道数
    in_channels = x.size(1)

    # 第一个门控卷积
    x1 = GatedConv2d(in_channels, n_filters, padding=1)(x)
    x2 = nn.BatchNorm2d(n_filters)(x1)
    x3 = F.relu(x2)
    x4 = nn.Dropout2d(dropout_rate)(x3)

    # 第二个门控卷积
    x5 = GatedConv2d(n_filters, n_filters, padding=1)(x4)
    x6 = nn.BatchNorm2d(n_filters)(x5)
    x7 = nn.Dropout2d(dropout_rate)(x6)

    # 残差连接
    if in_channels != n_filters:
        # 如果通道数不匹配，使用1x1卷积进行投影
        residual_conv = nn.Conv2d(in_channels, n_filters, kernel_size=1).to(x.device)
        identity = residual_conv(x)
    else:
        identity = x

    x8 = identity + x7

    # 最大池化
    x = nn.MaxPool2d(pool_size)(x8)

    # 最终激活
    x = F.relu(x)

    return x


# 为了保持与Keras版本相同的接口，创建一个包装类
class CRAMBlock(nn.Module):
    """CRAM Block module for easier integration in sequential models."""

    def __init__(self, in_channels: int, n_filters: int = 64,
                 pool_size: Tuple[int, int] = (2, 2),
                 dropout_rate: float = 0.2):
        super(CRAMBlock, self).__init__()

        self.n_filters = n_filters
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate

        # 第一个门控卷积路径
        self.gated_conv1 = GatedConv2d(in_channels, n_filters, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # 第二个门控卷积路径
        self.gated_conv2 = GatedConv2d(n_filters, n_filters, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # 残差连接
        if in_channels != n_filters:
            self.residual_conv = nn.Conv2d(in_channels, n_filters, kernel_size=1)
        else:
            self.residual_conv = None

        # 最大池化层
        self.maxpool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CRAM block."""
        identity = x

        # 第一个门控卷积路径
        x1 = self.gated_conv1(x)
        x2 = self.bn1(x1)
        x3 = F.relu(x2)
        x4 = self.dropout1(x3)

        # 第二个门控卷积路径
        x5 = self.gated_conv2(x4)
        x6 = self.bn2(x5)
        x7 = self.dropout2(x6)

        # 残差连接
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)

        x8 = identity + x7

        # 最大池化
        x = self.maxpool(x8)

        # 最终激活
        x = F.relu(x)

        return x


# 为了与原始Keras代码保持相同的调用方式，提供函数别名
block = cram_block
