import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union


class CapsuleLayer(nn.Module):
    """A PyTorch module implementing capsule routing [1].

    Args:
        n_capsules (int): Number of output capsules.
        dim_capsule (int): Number of units per output capsule.
        n_input_capsules (int): Number of input capsules.
        dim_input_capsule (int): Number of units per input capsule.
        routings (int): Number of routing iterations.
        use_bias (bool): Whether to use a bias vector.
        **kwargs: Other module arguments.

    References:
        [1] S. Sabour, N. Frosst, and G. E. Hinton, "Dynamic routing
            between capsules," in Adv. Neural Inf. Process. Syst.
            (NIPS), Long Beach, CA, 2017, pp. 3859–3869.
    """

    def __init__(self, n_capsules: int, dim_capsule: int, n_input_capsules: int,
                 dim_input_capsule: int, routings: int = 3, use_bias: bool = False,
                 **kwargs):
        super(CapsuleLayer, self).__init__()

        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule
        self.n_input_capsules = n_input_capsules
        self.dim_input_capsule = dim_input_capsule
        self.routings = routings
        self.use_bias = use_bias

        # Weight matrix: (n_capsules, n_input_capsules, dim_capsule, dim_input_capsule)
        self.W = nn.Parameter(
            torch.randn(n_capsules, n_input_capsules, dim_capsule, dim_input_capsule)
        )
        nn.init.xavier_uniform_(self.W)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_capsules))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply transformation followed by capsule routing."""
        batch_size = inputs.size(0)

        # 使用矩阵乘法计算预测向量
        # inputs: (batch, n_input, dim_input) -> (batch, 1, n_input, dim_input, 1)
        inputs_reshaped = inputs.unsqueeze(1).unsqueeze(4)

        # W: (n_caps, n_input, dim_caps, dim_input) -> (1, n_caps, n_input, dim_caps, dim_input)
        W_reshaped = self.W.unsqueeze(0)

        # 矩阵乘法: (batch, n_caps, n_input, dim_caps, 1)
        inputs_hat_temp = torch.matmul(W_reshaped, inputs_reshaped)

        # 压缩最后一个维度: (batch, n_caps, n_input, dim_caps)
        inputs_hat = inputs_hat_temp.squeeze(-1)

        # 添加偏置
        if self.use_bias and self.bias is not None:
            inputs_hat = inputs_hat + self.bias.view(1, -1, 1, 1)

        # 初始化路由logits
        b = torch.zeros(batch_size, self.n_capsules, self.n_input_capsules,
                        device=inputs.device)

        # 动态路由
        for i in range(self.routings):
            # 计算耦合系数
            c = F.softmax(b, dim=1)  # (batch, n_caps, n_input)

            # 计算输出胶囊
            weighted_inputs = c.unsqueeze(-1) * inputs_hat  # (batch, n_caps, n_input, dim_caps)
            outputs = squash(torch.sum(weighted_inputs, dim=2))  # (batch, n_caps, dim_caps)

            # 更新logits（除了最后一次迭代）
            if i < self.routings - 1:
                # 计算一致性（点积）
                outputs_expanded = outputs.unsqueeze(2)  # (batch, n_caps, 1, dim_caps)
                agreement = torch.sum(outputs_expanded * inputs_hat, dim=-1)  # (batch, n_caps, n_input)
                b = b + agreement

        return outputs


class PrimaryCapsules(nn.Module):
    """Apply a convolution followed by a squashing function.

    Args:
        in_channels (int): Number of input channels.
        n_channels (int): Number of channels per capsule.
        dim_capsule (int): Number of activation units per capsule.
        kernel_size (int or tuple): Size of convolution kernel.
        stride (int or tuple): Stride of convolution.
        padding (int or tuple): Padding for convolution.
        **kwargs: Other convolution arguments.
    """

    def __init__(self, in_channels: int, n_channels: int, dim_capsule: int,
                 kernel_size: Union[int, Tuple] = (3, 3), stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, **kwargs):
        super(PrimaryCapsules, self).__init__()

        self.n_channels = n_channels
        self.dim_capsule = dim_capsule

        self.conv = nn.Conv2d(
            in_channels, n_channels * dim_capsule, kernel_size,
            stride=stride, padding=padding, **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, n_capsules, dim_capsule)
        """
        # Apply convolution
        x = self.conv(x)  # (batch_size, n_channels * dim_capsule, height, width)

        # Reshape: (batch_size, n_capsules, dim_capsule)
        batch_size, _, height, width = x.shape
        n_capsules = self.n_channels * height * width

        x = x.view(batch_size, self.n_channels, self.dim_capsule, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, n_capsules, self.dim_capsule)

        # Apply squashing function
        return squash(x)


def squash(x: Tensor, dim: int = -1) -> Tensor:
    """Apply a squashing nonlinearity as described in [1].

    Args:
        x (Tensor): Input tensor to transform.
        dim (int): Axis along which squashing is applied.

    Returns:
        Tensor: Squashed output tensor.
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-7)
    return scale * x


def capsule_length(x: Tensor) -> Tensor:
    """Compute the Euclidean lengths of capsules.

    Args:
        x (Tensor): Tensor of capsules.

    Returns:
        Tensor: Euclidean lengths of capsules.
    """
    return torch.sqrt((x ** 2).sum(dim=-1))

