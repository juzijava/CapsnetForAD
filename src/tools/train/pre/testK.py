import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# 导入你 0.64 版本的模型定义
from src.network.hundernet  import FanAutoEncoder


def evaluate_k_range():
    # --- 1. 配置参数 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_car_best_auc.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_car_pro.h5"
    save_fig_path = "D:/tools/Pycode/CapsnetForAD/workresult/k_sensitivity_analysis.png"

    # --- 2. 加载模型 ---
    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f">>> 已成功加载模型: {model_path}")

    # --- 3. 提取测试集重构误差与潜空间特征 ---
    raw_scores = []
    all_features = []

    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    print(">>> 正在提取测试集特征...")
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            sample = X_test[i:i + 1].to(device)
            # mask_ratio 设为 0，保证测试的确定性
            recon, feat = model(sample, mask_ratio=0.0)

            # 计算重构误差 (Top 2% 策略，与你 0.64 版本一致)
            diff = (recon - sample) ** 2
            mse_per_frame = torch.mean(diff, dim=2).cpu().numpy()[0]
            top_k = max(1, int(len(mse_per_frame) * 0.02))
            raw_scores.append(np.sort(mse_per_frame)[-top_k:].mean())

            # 提取胶囊层特征作为 LDN 的邻域基准
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    raw_scores = np.array(raw_scores)
    all_features = np.array(all_features)

    # --- 4. 扫描 K 值 (从 3 到 100) ---
    k_list = [2,3, 4, 5, 6, 7,8 ,9, 10, ]
    auc_results = []

    print(f"\n{'K Value':<10} | {'AUC Score':<10}")
    print("-" * 25)

    for k in k_list:
        # 使用余弦距离寻找邻居
        knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(all_features)
        _, indices = knn.kneighbors(all_features)

        ldn_scores = []
        for i in range(len(raw_scores)):
            neighbor_indices = indices[i][1:]  # 剔除自身
            neighbor_avg = np.mean(raw_scores[neighbor_indices])
            ldn_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))

        auc = roc_auc_score(Y_test, ldn_scores)
        auc_results.append(auc)
        print(f"{k:<10} | {auc:<10.4f}")

    # --- 5. 绘制论文级图表 ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, auc_results, marker='s', markersize=8, linestyle='-', color='#2c3e50', linewidth=2,
             label='Proposed Model')

    max_auc = max(auc_results)
    best_k = k_list[auc_results.index(max_auc)]

    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.5)
    plt.annotate(f'Best AUC: {max_auc:.4f}\nK={best_k}',
                 xy=(best_k, max_auc), xytext=(best_k + 10, max_auc - 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, fontweight='bold')

    plt.title('Hyperparameter Sensitivity: Neighborhood Size K', fontsize=14)
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(min(auc_results) - 0.05, max(auc_results) + 0.05)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    print(f"\n>>> 敏感性分析图已保存至: {save_fig_path}")
    plt.show()


if __name__ == "__main__":
    evaluate_k_range()