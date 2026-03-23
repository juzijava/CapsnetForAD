import librosa
import numpy as np
import h5py
import os
from tqdm import tqdm


def extract_features_with_domain(audio_dir, output_file):
    """
    提取特征并记录 Domain 信息
    - labels: 0 为正常, 1 为异常
    - domains: 0 为 Source, 1 为 Target
    """
    sr, n_mels, n_fft, hop_length = 16000, 128, 2048, 512
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    features_list = []
    labels = []
    domains = []

    print(f">>> 正在提取特征并标记 Domain: {audio_dir}")

    for file_name in tqdm(audio_files):
        file_path = os.path.join(audio_dir, file_name)

        # 1. 自动解析标签和域
        # DCASE 文件名规范通常包含 'source' 或 'target'
        is_anomaly = 1 if 'anomaly' in file_name.lower() else 0
        is_target = 1 if 'target' in file_name.lower() else 0

        labels.append(is_anomaly)
        domains.append(is_target)

        # 2. 音频处理
        y, _ = librosa.load(file_path, sr=sr)
        # 预滤波 (可选，保持和你之前一致)
        y_filt = librosa.effects.preemphasis(y)

        # Mel 谱
        S = librosa.feature.melspectrogram(y=y_filt, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.median)

        # Delta 特征 (捕捉机械结构变化)
        delta1 = librosa.feature.delta(log_S)
        delta2 = librosa.feature.delta(log_S, order=2)

        # 拼接 [T, 384]
        combined = np.concatenate([log_S, delta1, delta2], axis=0).T

        # 维持你 0.6119 成功经验：不使用逐帧归一化，只进行全局缩放
        combined = combined / 10.0
        features_list.append(combined.astype(np.float32))

    # 3. 写入 H5 文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f_h5:
        f_h5.create_dataset('F_fan', data=np.array(features_list))
        f_h5.create_dataset('labels', data=np.array(labels))
        f_h5.create_dataset('domains', data=np.array(domains))  # 新增：0=Source, 1=Target

    print(f">>> 特征提取完成，已保存至: {output_file}")
    print(f"统计: Source 样本 {domains.count(0)} 个, Target 样本 {domains.count(1)} 个")


if __name__ == "__main__":
    # 请根据实际路径修改
    train_dir = r"/data/ToyCar/train"
    test_dir = r"/data/ToyCar/test"

    # 提取测试集（测试集通常包含 Source 和 Target 的混合）
    extract_features_with_domain(test_dir, "D:/tools/Pycode/CapsnetForAD/workresult/test_car_nomon_with_domain.h5")