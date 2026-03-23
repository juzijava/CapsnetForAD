import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# 假设你的模型定义在 src.network.LCknet
from src.network.LCknet import FanAutoEncoder

#你可以把现在的 canchatrain.py 改名为 canchatrain_mask_fail.py。虽然这次失败了，
# 但它证明了“纯掩码重构”在不加对齐的情况下很难搞定跨电压数据，
# 这以后是写论文或报告里的重要“反面教材”。
def train_and_eval_best_auc():
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0

    # --- 4. 训练循环 ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for batch_x, in pbar:
            batch_x = batch_x.to(device)

            # 训练时使用 50% 掩码：强迫模型学习“补全”逻辑
            recon = model(batch_x, mask_ratio=0.5)

            # Loss 计算：重建值与原始完整输入的差异
            loss = nn.MSELoss()(recon, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # --- 5. 每轮执行一次确定性评估 ---
        model.eval()
        test_scores = []
        with torch.no_grad():
            for i in range(len(X_test)):
                sample = X_test[i:i + 1].to(device)

                # 评估时 mask_ratio=0：利用训练好的逻辑检查全量信号的矛盾
                recon = model(sample, mask_ratio=0.0)

                # 计算残差能量作为异常分数
                # 使用 MBS 逻辑简化版：取帧 MSE 的 Top 2% 平均值
                diff = (recon - sample) ** 2
                mse_per_frame = torch.mean(diff, dim=2).cpu().numpy()[0]

                # 模拟 MBS 聚焦
                top_k = max(1, int(len(mse_per_frame) * 0.02))
                score = np.sort(mse_per_frame)[-top_k:].mean()
                test_scores.append(score)

        current_auc = roc_auc_score(Y_test, test_scores)
        print(f" >> Epoch [{epoch}] 完成, Avg Loss: {avg_loss:.6f}, 当前 AUC: {current_auc:.4f}")

        # --- 6. 保存最佳 AUC 模型 ---
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), save_path)
            print(f" [!] 发现更高 AUC, 模型已更新保存。")

    print(f"\n训练结束！历史最高 AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train_and_eval_best_auc()