import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os

# 导入修改后的模型
from src.network.hundernet import FanAutoEncoder


# ... 前面的导入代码保持不变 ...

class DynamicWeightedLoss(nn.Module):
    def __init__(self, device):
        super(DynamicWeightedLoss, self).__init__()
        self.device = device
        # 注册为 buffer，这样它会随模型移动到同一设备，且不会被当做参数更新
        self.register_buffer('base_weights', torch.ones(384))

    def forward(self, recon, target, epoch):
        # 1. 克隆基础权重
        weights = self.base_weights.clone()

        # 2. 计算动态权重
        # 0-128 维 (Log-Mel)
        weights[:128] = 0.05
        # 128-384 维 (Delta)：随 epoch 动态增强
        dynamic_factor = 2.0 + (epoch * 0.5) #toytrain
        # dynamic_factor = 5.0 + (epoch * 1.5)  # toycar
        weights[128:] = dynamic_factor

        # 3. 计算平方误差
        sq_err = (recon - target) ** 2

        # 【关键修复】：确保 weights 移动到与 sq_err 相同的设备 (如 GPU)
        weights = weights.to(sq_err.device)

        # 4. 应用权重并求平均
        weighted_err = sq_err * weights.view(1, 1, -1)
        return weighted_err.mean()


def train_and_eval_robust():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 调整参数以应对过拟合 ---
    batch_size = 16
    epochs = 100
    learning_rate = 1e-4  # 降低学习率，从 1e-3 降到 1e-4，让模型慢慢磨特征
    mask_ratio = 0.85  # 提高掩码率，让模型更难通过“背诵”来降低 Loss

    save_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_toytrain_nomon_best.pth"

    # 加载数据 (假设你已经注释掉了归一化并重新生成了 h5)
    print(">>> 正在加载训练与测试数据...")
    with h5py.File("D:/tools/Pycode/CapsnetForAD/workresult/train_toytrain_nomon.h5", 'r') as f:
        X_train = torch.from_numpy(f['F_fan'][:]).float()
    with h5py.File("D:/tools/Pycode/CapsnetForAD/workresult/test_toytrain_nomon.h5", 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = DynamicWeightedLoss(device)

    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for batch in pbar:
            batch_x = batch[0].to(device)

            # 使用更高的掩码率
            recon, _ = model(batch_x, mask_ratio=mask_ratio)

            # 【修复位置】：传入 epoch 变量
            loss = criterion(recon, batch_x, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 评估逻辑 (保持之前的 Mean 策略)
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            raw_scores = []
            all_features = []
            with torch.no_grad():
                for i in range(len(X_test)):
                    sample = X_test[i:i + 1].to(device)
                    recon, feat = model(sample, mask_ratio=0.0)

                    # 评估时也使用当前 epoch 的权重进行对齐
                    diff = (recon - sample) ** 2
                    # 也可以尝试使用更简单的评估权重：weights[:128]=0.1, weights[128:]=5.0
                    weighted_diff = diff * criterion.base_weights.to(device).view(1, 1, -1)

                    score = torch.mean(weighted_diff).item()
                    raw_scores.append(score)
                    all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

            # LDN 计算
            all_features = np.array(all_features)
            raw_scores = np.array(raw_scores)
            knn = NearestNeighbors(n_neighbors=15, metric='cosine')
            knn.fit(all_features)
            _, indices = knn.kneighbors(all_features)

            ldn_scores = []
            for i in range(len(raw_scores)):
                neighbor_avg = np.mean(raw_scores[indices[i][1:]])
                ldn_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))

            current_auc = roc_auc_score(Y_test, ldn_scores)
            print(f" >> [评估] Epoch {epoch}: LDN AUC = {current_auc:.4f}")

            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(model.state_dict(), save_path)
                print(" [!] 发现更好的模型，已保存。")


if __name__ == "__main__":
    train_and_eval_robust()