import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from src.network.spnet import STgramAttentionCapsuleNet  # 导入上面的模型


def train_with_st_fusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = "D:/tools/Pycode/CapsnetForAD/workresult/spbest_stgram_model.pth"

    # 1. 加载 STgram 数据
    print(">>> 正在加载时频融合数据...")
    with h5py.File("D:/tools/Pycode/CapsnetForAD/workresult/train_stgram_data.h5", 'r') as f:
        X_s_train = torch.from_numpy(f['sgram'][:])
        X_w_train = torch.from_numpy(f['wave'][:])
    with h5py.File("D:/tools/Pycode/CapsnetForAD/workresult/test_stgram_data.h5", 'r') as f:
        X_s_test = torch.from_numpy(f['sgram'][:])
        X_w_test = torch.from_numpy(f['wave'][:])
        Y_test = f['labels'][:]

    train_loader = DataLoader(TensorDataset(X_s_train, X_w_train), batch_size=16, shuffle=True)

    model = STgramAttentionCapsuleNet(sgram_dim=128, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_auc = 0.0

    # 2. 训练循环
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/100]")
        for b_s, b_w in pbar:
            b_s, b_w = b_s.to(device), b_w.to(device)
            recon, _ = model(b_s, b_w, mask_ratio=0.5)
            loss = nn.MSELoss()(recon, b_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # --- 3. 间隔评估逻辑 ---
        if epoch <= 50:
            should_eval = (epoch % 10 == 0)
        elif epoch <= 80:
            should_eval = (epoch % 10 == 0)
        else:
            should_eval = True

        if should_eval:
            model.eval()
            raw_scores, all_features = [], []
            print(f" >> Epoch {epoch} 执行 LDN (K=4) 评估...")

            with torch.no_grad():
                for i in range(len(X_s_test)):
                    s, w = X_s_test[i:i + 1].to(device), X_w_test[i:i + 1].to(device)
                    recon, feat = model(s, w, mask_ratio=0.0)

                    # MBS 评分 (Top 2%)
                    diff = ((recon - s) ** 2).mean(dim=2).cpu().numpy()[0]
                    top_k = max(1, int(len(diff) * 0.02))
                    raw_scores.append(np.sort(diff)[-top_k:].mean())
                    # 潜空间特征均值
                    all_features.append(feat.mean(dim=1).cpu().numpy()[0])

            # LDN 归一化 (K=4)
            raw_scores = np.array(raw_scores)
            all_features = np.array(all_features)
            knn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(all_features)
            _, indices = knn.kneighbors(all_features)

            ldn_scores = [raw_scores[i] / (np.mean(raw_scores[indices[i][1:]]) + 1e-8) for i in range(len(raw_scores))]
            current_auc = roc_auc_score(Y_test, ldn_scores)
            print(f" >> Epoch {epoch} 结果: AUC = {current_auc:.4f}")

            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(model.state_dict(), save_path)
                print(f" [!] 发现更优模型，已保存。")

    print(f"\n训练完成！最佳 STgram-LDN AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train_with_st_fusion()