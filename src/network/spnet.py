import torch
import torch.nn as nn
from ncps.torch import CfC


class FusionGate(nn.Module):
    """
    SCI二区核心：注意力门控模块
    利用全局时间上下文来决定 Sgram 和 Tgram 的权重分配
    """

    def __init__(self, s_dim, t_dim, reduction=4):
        super(FusionGate, self).__init__()
        combined_dim = s_dim + t_dim
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局池化，捕捉音频段的全局特征
            nn.Conv1d(combined_dim, combined_dim // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(combined_dim // reduction, combined_dim, 1),
            nn.Sigmoid()  # 输出 0-1 的掩码权重
        )

    def forward(self, s_feat, t_feat):
        # 拼接特征
        combined = torch.cat([s_feat, t_feat], dim=1)
        # 计算注意力掩码
        mask = self.gate(combined)
        # 动态赋权融合
        return combined * mask


class STgramAttentionCapsuleNet(nn.Module):
    def __init__(self, sgram_dim=128, hidden_dim=32):
        super().__init__()

        # --- 1. 时域路径 (Tgram) ---
        self.tgram_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1024, stride=512, padding=512),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )

        # --- 2. 门控融合层 ---
        self.gate_fusion = FusionGate(sgram_dim, 64)

        # --- 3. 融合后提取层 ---
        self.fusion_conv = nn.Conv1d(sgram_dim + 64, 64, kernel_size=5, padding=2)
        self.lnn_enc = CfC(64, hidden_dim, batch_first=True)

        # --- 4. 胶囊层 (核心判别器) ---
        self.num_caps, self.caps_dim = 4, 16
        self.W_caps = nn.Parameter(torch.randn(self.num_caps, hidden_dim, self.caps_dim) * 0.05)

        # --- 5. 重构解码器 ---
        self.decoder_lnn = CfC(self.num_caps * self.caps_dim, hidden_dim, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, sgram_dim)

    def squash(self, x):
        mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        return (mag_sq / (1 + mag_sq)) * (x / torch.sqrt(mag_sq + 1e-8))

    def forward(self, sgram, wave, mask_ratio=0.5):
        b, t, d = sgram.shape

        # 维度适配
        if wave.dim() == 4:
            wave = wave.squeeze(2)
        elif wave.dim() == 2:
            wave = wave.unsqueeze(1)

        # 训练时掩码增强
        if self.training:
            mask = torch.bernoulli(torch.full(sgram.shape, 1 - mask_ratio)).to(sgram.device)
            sgram = sgram * mask

        # Tgram 特征提取与对齐
        t_feat = self.tgram_net(wave)
        t_feat = torch.nn.functional.interpolate(t_feat, size=t, mode='linear')

        # 门控注意力融合
        s_feat = sgram.transpose(1, 2)
        fused = self.gate_fusion(s_feat, t_feat)

        # 编码
        feat = torch.relu(self.fusion_conv(fused)).transpose(1, 2)
        out_enc, _ = self.lnn_enc(feat)

        # 胶囊空间投影
        u = out_enc.unsqueeze(2).repeat(1, 1, self.num_caps, 1)
        caps = self.squash(torch.matmul(u.unsqueeze(-2), self.W_caps).squeeze(-2))

        # 解码重构
        caps_flat = caps.view(b, t, -1)
        out_dec, _ = self.decoder_lnn(caps_flat)
        recon = self.decoder_out(out_dec)

        return recon, caps_flat