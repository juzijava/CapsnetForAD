import torch
import torch.nn as nn
from ncps.torch import CfC


class FanAutoEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=32):
        super().__init__()
        # Encoder: 提取局部与时序特征
        self.encoder_conv = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.lnn_enc = CfC(64, hidden_dim, batch_first=True)

        # Capsule Layer: 结构化压缩
        self.num_caps, self.caps_dim = 4, 16
        self.W_caps = nn.Parameter(torch.randn(self.num_caps, hidden_dim, self.caps_dim) * 0.05)

        # Decoder: 尝试还原 384 维原始特征
        self.decoder_lnn = CfC(self.num_caps * self.caps_dim, hidden_dim, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def squash(self, x):
        mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        return (mag_sq / (1 + mag_sq)) * (x / torch.sqrt(mag_sq + 1e-8))

    def forward(self, x):
        # x: [B, T, 384]
        b, t, d = x.shape

        # Encoding
        x_in = x.transpose(1, 2)
        feat = torch.relu(self.encoder_conv(x_in)).transpose(1, 2)
        out_enc, _ = self.lnn_enc(feat)  # [B, T, 128]

        u = out_enc.unsqueeze(2).repeat(1, 1, self.num_caps, 1)
        caps = self.squash(torch.matmul(u.unsqueeze(-2), self.W_caps).squeeze(-2))

        # Decoding (重建全序列)
        caps_flat = caps.view(b, t, -1)  # [B, T, 64]
        out_dec, _ = self.decoder_lnn(caps_flat)
        recon = self.decoder_out(out_dec)  # [B, T, 384]

        return recon

    def forward(self, x, mask_ratio=0.5):
        # x: [B, T, 384]
        b, t, d = x.shape

        # --- 1. 强力掩码机制 ---
        # 随机生成掩码，1表示保留，0表示遮盖
        # 我们遮住 50% 的特征维度，强迫模型“动脑子”
        if self.training:
            mask = torch.bernoulli(torch.full(x.shape, 1 - mask_ratio)).to(x.device)
        else:
            # 评估时使用固定掩码或不掩码，但为了保持分布一致，建议保留轻微遮盖
            mask = torch.bernoulli(torch.full(x.shape, 0.8)).to(x.device)

        x_masked = x * mask

        # --- 2. Encoding (基于残缺信号提取特征) ---
        x_in = x_masked.transpose(1, 2)
        feat = torch.relu(self.encoder_conv(x_in)).transpose(1, 2)
        out_enc, _ = self.lnn_enc(feat)

        u = out_enc.unsqueeze(2).repeat(1, 1, self.num_caps, 1)
        # 胶囊层在这里发挥作用：通过残缺特征推断整体“姿态”
        caps = self.squash(torch.matmul(u.unsqueeze(-2), self.W_caps).squeeze(-2))

        # --- 3. Decoding (预测补全量) ---
        caps_flat = caps.view(b, t, -1)
        out_dec, _ = self.decoder_lnn(caps_flat)
        delta = self.decoder_out(out_dec)

        # --- 4. 【关键修正】残差叠加 ---
        # 不要用完整的 x，要用被遮掉的 x_masked
        # 这样模型必须通过 delta 把被遮掉的那 50% 能量补回来
        recon = x_masked + delta

        return recon