import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# 假设已经有转换好的capsules.py和gated_conv.py
from src.models.capsules import PrimaryCapsules, capsule_length, CapsuleLayer
from src.models import gated_conv


class GCCaps(nn.Module):
    """PyTorch implementation of GCCaps architecture for audio classification.

    Args:
        input_shape (tuple): Shape of the input tensor (time_steps, freq_bins)
        n_classes (int): Number of classes for classification
    """

    def __init__(self, input_shape: Tuple[int, int], n_classes: int):
        super(GCCaps, self).__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes

        # 添加通道维度
        self.reshape = lambda x: x.unsqueeze(-1)

        # 三个门控卷积块
        self.gated_conv_blocks = nn.Sequential(
            gated_conv.CRAMBlock(in_channels=1, n_filters=64, pool_size=(2, 2)),
            gated_conv.CRAMBlock(in_channels=64, n_filters=64, pool_size=(2, 2)),
            gated_conv.CRAMBlock(in_channels=64, n_filters=64, pool_size=(2, 2))
        )

        # 计算经过卷积块后的特征图尺寸
        time_steps, freq_bins = input_shape
        # 三次池化，每次时间维度减半
        self.n_steps = time_steps // (2 * 2 * 2)  # 100 -> 12 (统一使用 n_steps)
        self.freq_after_conv = freq_bins // (2 * 2 * 2)  # 64 -> 8

        # 初级胶囊层
        self.primary_caps = PrimaryCapsules(
            in_channels=64,
            n_channels=16,
            dim_capsule=4,
            kernel_size=3,
            stride=(1, 2),  # 频率维度减半
            padding=1
        )

        # 计算每个时间步的胶囊数量
        freq_after_primary = self.freq_after_conv // 2  # stride=(1,2) 频率维度减半
        self.capsules_per_step = 16 * freq_after_primary  # 16 * 4 = 64

        # 时间分布胶囊层
        self.time_distributed_caps = TimeDistributedCapsule(
            n_capsules=n_classes,
            dim_capsule=8,
            routings=3,
            n_input_capsules=self.capsules_per_step
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)

        # 重塑输入: (batch, time, freq) -> (batch, time, freq, 1)
        x = x.unsqueeze(-1)

        # 调整维度顺序以匹配卷积: (batch, time, freq, 1) -> (batch, 1, time, freq)
        x = x.permute(0, 3, 1, 2)

        # 应用门控卷积块
        x = self.gated_conv_blocks(x)  # (batch, 64, n_steps, freq_after_conv)

        # 应用初级胶囊层
        x = self.primary_caps(x)  # (batch, n_capsules, 4)

        # 计算总胶囊数量
        n_primary_caps = x.size(1)  # 应该是 16 * n_steps * 4 = 768

        # 重塑为时间分布格式: (batch, n_steps, capsules_per_step, 4)
        x = x.view(batch_size, self.n_steps, self.capsules_per_step, 4)

        # 应用时间分布胶囊层
        caps = self.time_distributed_caps(x)  # (batch, n_steps, n_classes, 8)

        # 计算胶囊长度作为分类概率
        output = capsule_length(caps)  # (batch, n_steps, n_classes)

        # 沿时间维度平均
        output = output.mean(dim=1)  # (batch, n_classes)

        return output


class TimeDistributedCapsule(nn.Module):
    """Time-distributed capsule layer implementation.

    Args:
        n_capsules (int): Number of output capsules per time step
        dim_capsule (int): Dimension of each output capsule
        routings (int): Number of routing iterations
        n_input_capsules (int): Number of input capsules per time step
    """

    def __init__(self, n_capsules: int, dim_capsule: int, routings: int = 3, n_input_capsules: int = 16):
        super(TimeDistributedCapsule, self).__init__()

        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.n_input_capsules = n_input_capsules

        # 创建一个共享的胶囊层，用于所有时间步
        self.capsule_layer = CapsuleLayer(  # 注意这里是单数 capsule_layer
            n_capsules=n_capsules,
            dim_capsule=dim_capsule,
            n_input_capsules=n_input_capsules,
            dim_input_capsule=4,  # 初级胶囊的维度
            routings=routings
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply capsule layer to each time step.

        Args:
            x: Input tensor of shape (batch_size, n_steps, n_input_capsules, dim_input_capsule)

        Returns:
            Output tensor of shape (batch_size, n_steps, n_capsules, dim_capsule)
        """
        batch_size, n_steps, n_input_capsules, dim_input_capsule = x.shape

        # 为每个时间步应用胶囊层
        outputs = []
        for t in range(n_steps):
            # 获取当前时间步的输入
            x_t = x[:, t, :, :]  # (batch_size, n_input_capsules, dim_input_capsule)

            # 应用共享的胶囊层 - 修正：使用 capsule_layer 而不是 capsule_layers[0]
            caps_t = self.capsule_layer(x_t)  # (batch_size, n_capsules, dim_capsule)
            outputs.append(caps_t)

        # 堆叠所有时间步的输出
        output = torch.stack(outputs, dim=1)  # (batch_size, n_steps, n_capsules, dim_capsule)

        return output


def gccaps_predict(x: np.ndarray, model: GCCaps, batch_size: int = 32,
                   device: torch.device = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate output predictions for the given input examples.

    Args:
        x (np.ndarray): Array of input examples
        model (GCCaps): PyTorch model of GCCaps architecture
        batch_size (int): Number of examples in a mini-batch
        device: Device to run inference on

    Returns:
        tuple: A tuple containing the audio tagging predictions and SED predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # 转换为PyTorch张量
    x_tensor = torch.FloatTensor(x).to(device)

    # 分批处理
    n_samples = len(x)
    n_batches = (n_samples + batch_size - 1) // batch_size

    at_preds = []
    sed_preds = []

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_x = x_tensor[start_idx:end_idx]

            # 音频标签预测
            at_batch = model(batch_x)
            at_preds.append(at_batch.cpu().numpy())

            # 声音事件检测预测 - 需要访问中间层输出
            # 这里需要根据具体实现调整
            # 简化版本：使用胶囊层的输出作为SED预测
            # 实际实现可能需要hook机制来获取中间层输出
            pass

    # 合并批次结果
    at_preds = np.concatenate(at_preds, axis=0)

    return at_preds, None  # SED预测需要额外实现


class MergeLayer(nn.Module):
    """Merge layer for combining capsule outputs with attention weights."""

    def __init__(self, method: str = 'standard'):
        super(MergeLayer, self).__init__()
        self.method = method

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Merge capsule outputs with attention weights.

        Args:
            inputs: Tuple of (capsules, attention_weights)
                   capsules: (batch_size, time_steps, n_classes)
                   attention_weights: (batch_size, time_steps, n_classes)

        Returns:
            Merged output of shape (batch_size, n_classes)
        """
        caps, att = inputs

        if self.method == 'standard':
            # 标准合并方法
            att = torch.clamp(att, min=1e-7, max=1.0)
            return torch.sum(caps * att, dim=1) / torch.sum(att, dim=1)

        elif self.method == 'exp':
            # 指数加权合并方法
            exp_att = torch.exp(att)
            return torch.sum(caps * exp_att, dim=1) / torch.sum(exp_att, dim=1)

        else:
            raise ValueError(f"Unknown merge method: {self.method}")


# 创建模型的便捷函数
def create_gccaps(input_shape: Tuple[int, int], n_classes: int) -> GCCaps:
    """Create a GCCaps model with the given configuration.

    Args:
        input_shape (tuple): Shape of the input tensor (time_steps, freq_bins)
        n_classes (int): Number of output classes

    Returns:
        GCCaps model instance
    """
    return GCCaps(input_shape=input_shape, n_classes=n_classes)


# 测试代码
if __name__ == "__main__":
    # 测试模型
    input_shape = (100, 64)  # 100个时间步，64个频率bin
    n_classes = 10

    model = create_gccaps(input_shape, n_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # 测试前向传播
    batch_size = 4
    x_test = torch.randn(batch_size, *input_shape)
    output = model(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {output.shape}")