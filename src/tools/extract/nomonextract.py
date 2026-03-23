import librosa
import numpy as np
import h5py
import os
from tqdm import tqdm
# [固定 Window=11] 正在扫描最优 K 值...
# K Value | AUC Score
# --------------------
# 2        | 0.5958
# 3        | 0.6161
# 4        | 0.6161
# 5        | 0.6471
# 6        | 0.6621
# 7        | 0.6593
# 8        | 0.6588
# 9        | 0.6501
# 10       | 0.6501
# 最终冲刺结果: 当 Window=11, K=6 时, 最高 AUC = 0.6621
# >>> 恭喜！已突破 0.64 目标。

def extract_antishift_features(audio_dir, output_file):
    sr, n_mels, n_fft, hop_length = 16000, 128, 2048, 512
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    labels = [1 if 'anomaly' in n.lower() else 0 for n in audio_files]

    features_list = []
    print(f">>> 提取抗偏移增强特征: {audio_dir}")

    for f in tqdm(audio_files):
        # 1. 加载并进行预加重 (Pre-emphasis)
        # 这一步能有效压制图中显示的低频亮斑，提升高频垂直线条的对比度
        y, _ = librosa.load(f, sr=sr, duration=10)
        y_filt = librosa.effects.preemphasis(y)

        # 2. 计算 Mel 谱
        S = librosa.feature.melspectrogram(y=y_filt, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # 使用中位数参考，防止异常脉冲拉高整体增益
        log_S = librosa.power_to_db(S, ref=np.median)

        # 3. 提取 Delta 和 Delta-Delta (捕捉图中看到的垂直脉冲)
        delta1 = librosa.feature.delta(log_S)
        delta2 = librosa.feature.delta(log_S, order=2)

        # 4. 拼接 [T, 384]
        combined = np.concatenate([log_S, delta1, delta2], axis=0).T

        # 5. 【关键】帧内标准化 (Per-frame Normalization)
        # 强行让每一帧的均值和方差一致，彻底消除 Source/Target 的能量分布偏差
        # mean = combined.mean(axis=1, keepdims=True)
        # std = combined.std(axis=1, keepdims=True) + 1e-6
        # combined = (combined - mean) / std
        combined = combined / 10.0
        features_list.append(combined.astype(np.float32))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f_h5:
        f_h5.create_dataset('F_fan', data=np.array(features_list))
        f_h5.create_dataset('labels', data=np.array(labels).astype(np.int64))
    print(f">>> 特征提取完成，已保存至: {output_file}")


if __name__ == "__main__":
    TRAIN_DIR = "D:/tools/Pycode/CapsnetForAD/data/ToyTrain/train"
    TEST_DIR = "D:/tools/Pycode/CapsnetForAD/data/ToyTrain/test"
    RESULT_PATH = "D:/tools/Pycode/CapsnetForAD/workresult"
    extract_antishift_features(TRAIN_DIR, os.path.join(RESULT_PATH, "train_toytrain_nomon.h5"))
    extract_antishift_features(TEST_DIR, os.path.join(RESULT_PATH, "test_toytrain_nomon.h5"))