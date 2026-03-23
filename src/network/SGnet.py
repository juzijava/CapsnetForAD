import torch
import torch.nn as nn
from ncps.torch import CfC

class FanAutoEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=32):
        super().__init__()
        self.encoder_conv = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.lnn_enc = CfC(64, hidden_dim, batch_first=True)
        self.num_caps, self.caps_dim = 4, 16
        self.W_caps = nn.Parameter(torch.randn(self.num_caps, hidden_dim, self.caps_dim) * 0.05)
        self.decoder_lnn = CfC(self.num_caps * self.caps_dim, hidden_dim, batch_first=True)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def squash(self, x):
        mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        return (mag_sq / (1 + mag_sq)) * (x / torch.sqrt(mag_sq + 1e-8))

    def forward(self, x, mask_ratio=0.5):
        b, t, d = x.shape
        if self.training:
            mask = torch.bernoulli(torch.full(x.shape, 1 - mask_ratio)).to(x.device)
        else:
            mask = torch.ones_like(x).to(x.device)
        x_masked = x * mask

        x_in = x_masked.transpose(1, 2)
        feat = torch.relu(self.encoder_conv(x_in)).transpose(1, 2)
        out_enc, _ = self.lnn_enc(feat)
        u = out_enc.unsqueeze(2).repeat(1, 1, self.num_caps, 1)
        caps = self.squash(torch.matmul(u.unsqueeze(-2), self.W_caps).squeeze(-2))

        # 特征向量接口
        caps_flat = caps.view(b, t, -1)
        out_dec, _ = self.decoder_lnn(caps_flat)
        recon = self.decoder_out(out_dec)

        return recon, caps_flat