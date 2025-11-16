import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# 假设导入相关模块
from src.models.capsules import PrimaryCapsules, CapsuleLayer, capsule_length


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetFeatureExtractor(nn.Module):
    """基于ResNet的特征提取器"""

    def __init__(self, input_shape: Tuple[int, int]):
        super(ResNetFeatureExtractor, self).__init__()
        self.input_shape = input_shape

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        self.n_steps = 78
        self.capsules_per_step = 64

        # 初级胶囊层
        self.primary_caps = PrimaryCapsules(
            in_channels=128,
            n_channels=8,
            dim_capsule=4,
            kernel_size=3,
            stride=(1, 2),
            padding=1
        )

        self.time_distributed_caps = TimeDistributedCapsule(
            n_capsules=16,
            dim_capsule=8,
            routings=2,
            n_input_capsules=self.capsules_per_step
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建残差层"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.primary_caps(x)
        x = x.view(batch_size, self.n_steps, self.capsules_per_step, -1)

        caps = self.time_distributed_caps(x)

        return caps



class TimeDistributedCapsule(nn.Module):
    """时间分布胶囊层"""

    def __init__(self, n_capsules: int, dim_capsule: int, routings: int = 2, n_input_capsules: int = 16):
        super(TimeDistributedCapsule, self).__init__()

        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

        self.capsule_layer = CapsuleLayer(
            n_capsules=n_capsules,
            dim_capsule=dim_capsule,
            n_input_capsules=n_input_capsules,
            dim_input_capsule=4,
            routings=routings
        )

    def forward(self, x):
        batch_size, n_steps, n_input_capsules, dim_input = x.shape

        outputs = []
        for t in range(n_steps):
            x_t = x[:, t, :, :]
            caps_t = self.capsule_layer(x_t)
            outputs.append(caps_t)

        return torch.stack(outputs, dim=1)


class ResNetAnomalyDetectionCapsNet(nn.Module):
    """基于ResNet的异常检测胶囊网络"""

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape  # (311, 64)

        # ResNet特征提取器
        self.feature_extractor = ResNetFeatureExtractor(input_shape)

        # 编码器-解码器
        self.encoder = nn.Sequential(
            nn.Linear(16 * 8, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, input_shape[0] * input_shape[1]),
            nn.Tanh()  # 使用tanh控制输出范围
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        caps_features = self.feature_extractor(x)

        batch_size, n_steps, n_caps, cap_dim = caps_features.shape

        # 简化注意力机制
        caps_lengths = torch.norm(caps_features, dim=-1)
        attention_weights = F.softmax(caps_lengths, dim=1)
        weighted_caps = caps_features * attention_weights.unsqueeze(-1)
        compressed_caps = weighted_caps.sum(dim=1)

        # 编码-解码
        caps_flat = compressed_caps.reshape(batch_size, -1)
        encoded = self.encoder(caps_flat)
        reconstructed_flat = self.decoder(encoded)
        reconstructed = reconstructed_flat.view(batch_size, *self.input_shape)

        # 缩放重建输出以匹配输入范围
        reconstructed = reconstructed * 5.0  # 根据输入范围调整

        return reconstructed, caps_features

    def compute_anomaly_score(self, x):
        """基于重建误差计算异常分数"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            # 确保输入和重建的形状相同
            assert x.shape == reconstructed.shape, f"形状不匹配: 输入{x.shape} vs 重建{reconstructed.shape}"
            mse_loss = nn.MSELoss(reduction='none')(reconstructed, x)
            anomaly_score = mse_loss.mean(dim=(1, 2))
            return anomaly_score.cpu().numpy()



def train_resnet_capsnet(model, train_loader, val_loader, num_epochs, device, verbose=True):
    """训练ResNet胶囊网络"""
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    print(f"开始训练，共 {num_epochs} 轮，训练样本: {len(train_loader.dataset)}")
    print(f"优化器: AdamW, 学习率: 0.0005, 权重衰减: 1e-4")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            reconstructed, caps_features = model(data)

            loss = criterion(reconstructed, data)

            # 特征多样性正则化
            feature_diversity = caps_features.std()
            if feature_diversity < 0.01:
                loss = loss + 0.01 * (0.01 - feature_diversity)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # 显示训练进度
            if batch_idx % 10 == 0 and verbose:
                print(f'  批次 {batch_idx}/{len(train_loader)}, 损失: {loss.item():.6f}')

            # 打印初始训练信息
            if verbose and epoch == 0 and batch_idx == 0:
                print(f"初始训练信息:")
                print(f"  输入形状: {data.shape}")
                print(f"  特征范围: [{caps_features.min():.6f}, {caps_features.max():.6f}]")
                print(f"  输出范围: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
                print(f"  特征均值: {caps_features.mean():.6f}")
                print(f"  初始损失: {loss.item():.6f}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 改进的日志输出
        if verbose or (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.6f}, '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Patience: {patience_counter}/15')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if verbose:
                print(f"  ✓ 新的最佳验证损失: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= 15:
            if verbose:
                print(f"早停触发，在 epoch {epoch + 1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"加载最佳模型，最终验证损失: {best_val_loss:.6f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'model': model
    }


# 创建模型的便捷函数
def create_resnet_capsnet(input_shape: Tuple[int, int]):
    return ResNetAnomalyDetectionCapsNet(input_shape=input_shape)
