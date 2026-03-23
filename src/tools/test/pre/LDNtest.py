import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def run_final_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/workresult/model_toycar_nomon_best.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_car_nomon_with_domain.h5"

    # 1. 加载模型
    from src.network.hundernet import FanAutoEncoder
    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 读取数据
    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]
        D_test = f['domains'][:]

    # 3. 推理 (确保权重为全 1.0)
    raw_scores, all_features = [], []
    eval_weights = torch.ones(384).to(device)

    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)
            # 计算重构误差
            diff = ((recon - sample) ** 2) * eval_weights.view(1, 1, -1)
            raw_scores.append(torch.mean(diff).item())
            # 提取特征用于 LDN
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    raw_scores = np.array(raw_scores)
    all_features = np.array(all_features)

    # 4. 执行 LDN
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(all_features)
    _, indices = knn.kneighbors(all_features)
    ldn_scores = np.array(
        [raw_scores[i] / (np.mean(raw_scores[indices[i][1:]]) + 1e-8) for i in range(len(raw_scores))])

    # 5. 计算并打印结果
    print("\n" + "=" * 55)
    print(f"{'Domain / Metric':<20} | {'AUC (LDN)':<15} | {'pAUC (LDN)':<15}")
    print("-" * 55)

    def get_stats(mask):
        y, s = Y_test[mask], ldn_scores[mask]
        auc = roc_auc_score(y, s)
        pauc = roc_auc_score(y, s, max_fpr=0.1)
        return auc, pauc

    # 全局
    o_auc, o_pauc = get_stats(np.ones_like(Y_test, dtype=bool))
    print(f"{'Overall':<20} | {o_auc:.4f}         | {o_pauc:.4f}")

    # Source (Domain 0)
    s_auc, s_pauc = get_stats(D_test == 0)
    print(f"{'Source Domain':<20} | {s_auc:.4f}         | {s_pauc:.4f}")

    # Target (Domain 1)
    t_auc, t_pauc = get_stats(D_test == 1)
    print(f"{'Target Domain':<20} | {t_auc:.4f}         | {t_pauc:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    run_final_eval()