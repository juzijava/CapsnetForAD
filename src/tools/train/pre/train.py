import torch
import torch.nn as nn
import h5py
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.network.LCknet import FanAutoEncoder  # 请确保模型定义文件路径正确

# 解决 OpenMP 冲突警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main_train_task():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 配置路径
    train_file = "/workresult/train_car_pro.h5"
    test_file = "/workresult/test_car_pro.h5"
    save_path = "/workresult/model_car_best.pth"  # 模型保存路径

    print(">>> 正在加载数据...")
    with h5py.File(train_file, 'r') as f:
        X_train = torch.from_numpy(f['F_fan'][:]).float()
        Y_train = f['labels'][:]
    with h5py.File(test_file, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    # 只取正常数据进行自监督训练
    train_loader = DataLoader(TensorDataset(X_train[Y_train == 0]), batch_size=16, shuffle=True)

    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 2. 训练阶段：带掩码的自监督学习
    print(">>> 正在进行掩码自监督重建训练 (学习正常机器逻辑)...")
    model.train()
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for (batch_x,) in pbar:
            batch_x = batch_x.to(device)

            # --- 核心改进：高强度掩码 ---
            # 遮掉 50% 的特征，逼迫模型从极少量信息中“脑补”规律
            mask = (torch.rand_like(batch_x) > 0.5).float().to(device)
            masked_x = batch_x * mask

            recon = model(masked_x)
            loss = criterion(recon, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f" >> Epoch [{epoch + 1}/{epochs}] 完成, 平均 Loss: {avg_loss:.6f}")

    # --- 新增：保存训练好的模型 ---
    torch.save(model.state_dict(), save_path)
    print(f"\n>>> 训练完成！模型已保存至: {save_path}")

    # 3. 评估阶段：引入终极 MBS 逻辑
    model.eval()
    test_scores = []
    print("\n>>> 正在执行 MBS (中值背景抑制) 异常评估...")

    with torch.no_grad():
        for i in tqdm(range(len(X_test)), desc="Evaluating"):
            sample = X_test[i:i + 1].to(device)
            recon = model(sample)

            # 策略 A: 频率解耦 - 只看 Delta 特征 (128-384维)
            # 屏蔽电压导致的基频偏移
            diff = (sample[:, :, 128:] - recon[:, :, 128:]) ** 2
            mse_per_frame = torch.mean(diff, dim=2).cpu().numpy()[0]

            # 1. 引入时序平滑 (Moving Average)
            # 异常冲击通常持续几个时间帧，平滑可以抑制随机噪声
            mse_smooth = np.convolve(mse_per_frame, np.ones(5) / 5, mode='same')

            # 2. MBS 标准化
            baseline = np.median(mse_smooth)
            mad = np.median(np.abs(mse_smooth - baseline)) + 1e-6
            robust_scores = (mse_smooth - baseline) / mad
            robust_scores = np.maximum(0, robust_scores)

            # 3. 峰值非线性放大 (Power Scoring)
            # 让高的更高，低的更低
            robust_scores = np.power(robust_scores, 2)

            # 4. 极度聚焦：只取前 2% 的最强信号
            top_k_count = max(1, int(len(robust_scores) * 0.02))
            final_score = np.sort(robust_scores)[-top_k_count:].mean()
            test_scores.append(final_score)

    # 计算最终 AUC
    auc = roc_auc_score(Y_test, test_scores)
    print(f"\n" + "=" * 45)
    print(f"DNDR-Net (MBS版) 评估完成")
    print(f"最终评估 AUC: {auc:.4f}")
    print(f"=" * 45)


if __name__ == "__main__":
    main_train_task()