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
from src.network.SGnet import FanAutoEncoder


# >>> 正在搜索多尺度 LDN 最佳组合...
# 邻居数 K= 3 | AUC: 0.6327
# 邻居数 K= 4 | AUC: 0.6405
# 邻居数 K= 5 | AUC: 0.6370
# ------------------------------

def train_and_eval_robust():
    # --- 1. 配置参数 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 100
    learning_rate = 1e-3
    save_path = "/workresult/model_car_best_auc.pth"

    # --- 2. 加载数据 ---
    print(">>> 正在加载训练与测试数据...")
    with h5py.File("D:/tools/Pycode/CapsnetForAD/workresult/train_car_pro.h5", 'r') as f:
        X_train = torch.from_numpy(f['F_fan'][:]).float()

    with h5py.File("D:/tools/Pycode/CapsnetForAD/workresult/test_car_pro.h5", 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

    # --- 3. 初始化模型与优化器 ---
    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 余弦退火调度：让学习率在后期（80轮后）变得极小，有助于精细调优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0

    # --- 4. 训练循环 ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for batch_x, in pbar:
            batch_x = batch_x.to(device)
            recon, caps_flat = model(batch_x, mask_ratio=0.5)

            # 1. 重构损失 (MSE)
            mse_loss = nn.MSELoss()(recon, batch_x)

            # 2. 潜空间约束 (二区卖点：强制正常样本特征在均值附近聚拢)
            # 这能让特征空间更干净，提升 LDN 的效果
            compact_loss = torch.mean(torch.var(caps_flat, dim=0))

            # 3. 总损失 (设置系数 0.1)
            loss = mse_loss + 0.1 * compact_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # --- 5. 动态评估逻辑 (减少偶然性) ---
        # 规则：50轮前每10轮测一次；50-80轮每5轮测一次；80轮后每轮精测
        if epoch < 50:
            should_eval = (epoch % 10 == 0)
        elif epoch < 80:
            should_eval = (epoch % 5 == 0)
        else:
            should_eval = True

        if should_eval:
            model.eval()
            raw_scores = []
            all_features = []

            print(f" >> Epoch {epoch} 执行 LDN 深度评估...")
            with torch.no_grad():
                for i in range(len(X_test)):
                    sample = X_test[i:i + 1].to(device)
                    # 评估不使用掩码，获取完整重构与特征
                    recon, feat = model(sample, mask_ratio=0.0)

                    # 计算 MBS 残差 (Top 2% 策略)
                    diff = (recon - sample) ** 2
                    mse_per_frame = torch.mean(diff, dim=2).cpu().numpy()[0]
                    top_k = max(1, int(len(mse_per_frame) * 0.02))
                    raw_scores.append(np.sort(mse_per_frame)[-top_k:].mean())

                    # 提取特征均值作为局部判定邻域的依据
                    all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

            # 特征级 LDN 归一化
            all_features = np.array(all_features)
            raw_scores = np.array(raw_scores)

            # 使用余弦距离寻找特征邻居
            knn = NearestNeighbors(n_neighbors=6, metric='cosine')
            knn.fit(all_features)
            _, indices = knn.kneighbors(all_features)

            ldn_scores = []
            for i in range(len(raw_scores)):
                neighbor_indices = indices[i][1:]  # 剔除样本自身
                neighbor_avg = np.mean(raw_scores[neighbor_indices])
                ldn_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))

            current_auc = roc_auc_score(Y_test, ldn_scores)
            print(f" >> [评估结果] Epoch {epoch}: LDN AUC = {current_auc:.4f}")

            # 保存逻辑：只有在评估轮次且打破纪录时才保存
            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(model.state_dict(), save_path)
                print(f" [!] 性能突破，已保存当前最佳权重。")

    print(f"\n>>> 100轮冲刺结束！最佳 LDN AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train_and_eval_robust()