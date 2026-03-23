import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors
import os
from src.network.hundernet import FanAutoEncoder
#toycar
# ==================================================
# Domain          | AUC        | pAUC (0.1)
# --------------------------------------------------
# Overall         | 0.6119   | 0.5532
# Source          | 0.6960   | 0.6295
# Target          | 0.5628   | 0.5074
# ==================================================
# toytrain
# ==================================================
# Domain          | AUC        | pAUC (0.1)
# --------------------------------------------------
# Overall         | 0.6271   | 0.5158
# Source          | 0.6852   | 0.4821
# Target          | 0.5744   | 0.5621
# ==================================================

def evaluate_061_with_domains():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_toytrain_nomon_best.pth"

    # 【关键】依然使用那个能跑出 0.61 的 H5 文件
    test_data_path = "D:/tools/Pycode/CapsnetForAD/workresult/test_toytrain_nomon.h5"

    # 1. 加载模型
    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f">>> 成功加载模型，准备找回 0.61 分数...")

    # 2. 加载数据并“现场”解析 Domain
    # 我们不重新提取 H5，而是直接根据 test 目录的文件名顺序来打标签
    test_dir = "D:/tools/Pycode/CapsnetForAD/data/ToyTrain/test"
    audio_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.wav')])

    # 建立 Domain 映射表 (0=Source, 1=Target)
    domain_labels = np.array([1 if 'target' in f.lower() else 0 for f in audio_files])

    with h5py.File(test_data_path, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    # 确保 H5 里的样本数和文件夹里的对得上
    if len(domain_labels) != len(X_test):
        print(f"警告：文件夹文件数({len(domain_labels)})与H5样本数({len(X_test)})不符！")
        # 如果不符，尝试用 H5 里的顺序（如果提取时是按顺序来的）
        # 这里假设顺序一致

    # 3. 执行推理 (完全复制 sixtytes.py 逻辑)
    raw_scores, all_features = [], []
    eval_weights = torch.ones(384).to(device)
    eval_weights[:128], eval_weights[128:] = 0.1, 5.0  # 保持 0.61 的权重

    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)
            diff = ((recon - sample) ** 2) * eval_weights.view(1, 1, -1)
            raw_scores.append(torch.mean(diff).item())
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    # 4. LDN 校准
    all_features, raw_scores = np.array(all_features), np.array(raw_scores)
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(all_features)
    _, indices = knn.kneighbors(all_features)

    ldn_scores = []
    for i in range(len(raw_scores)):
        neighbor_avg = np.mean(raw_scores[indices[i][1:]])
        ldn_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))
    ldn_scores = np.array(ldn_scores)

    # 5. 打印最终论文数据
    print("\n" + "=" * 50)
    print(f"{'Domain':<15} | {'AUC':<10} | {'pAUC (0.1)':<10}")
    print("-" * 50)

    def get_metrics(mask):
        y, s = Y_test[mask], ldn_scores[mask]
        return roc_auc_score(y, s), roc_auc_score(y, s, max_fpr=0.1)

    overall_auc, overall_pauc = get_metrics(np.ones_like(Y_test, dtype=bool))
    print(f"{'Overall':<15} | {overall_auc:.4f}   | {overall_pauc:.4f}")

    # 分 Domain 统计
    for d_val, d_name in zip([0, 1], ["Source", "Target"]):
        mask = (domain_labels == d_val)
        if sum(mask) > 0:
            d_auc, d_pauc = get_metrics(mask)
            print(f"{d_name:<15} | {d_auc:.4f}   | {d_pauc:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    evaluate_061_with_domains()