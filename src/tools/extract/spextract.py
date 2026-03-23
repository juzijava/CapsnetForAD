import os
import numpy as np
import librosa
import h5py
from tqdm import tqdm


def extract_st_features(data_dir, output_h5, sample_rate=16000, n_mels=128, frame_hop=512):
    """
    提取时频融合特征所需的原始数据
    data_dir: 音频文件存放路径 (包含 normal/anomaly 子文件夹)
    """
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

    # 准备存储空间
    all_sgram = []
    all_wave = []
    all_labels = []

    print(f">>> 开始提取特征，目标文件数: {len(file_list)}")

    for file_path in tqdm(file_list):
        # 1. 加载音频
        y, sr = librosa.load(file_path, sr=sample_rate)

        # 标签处理 (假设文件名包含 'normal' 或 'anomaly')
        label = 1 if 'anomaly' in file_path else 0

        # 2. 提取 Log-Mel 频谱 (频域轨迹)
        # 论文建议使用较细的帧移来保持时间分辨率
        sgram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=frame_hop)
        sgram_db = librosa.power_to_db(sgram, ref=np.max).T  # [T, 128]

        # 3. 提取原始波形切片 (时域轨迹)
        # 为了与频谱对齐，我们将波形进行简单的归一化
        # 论文中的 Tgram 分支会在模型内部处理这个长向量
        wave = y.reshape(1, -1)  # [1, L]

        all_sgram.append(sgram_db.astype(np.float32))
        all_wave.append(wave.astype(np.float32))
        all_labels.append(label)

    # 4. 保存为 H5 文件
    with h5py.File(output_h5, 'w') as f:
        # 注意：由于音频长度可能不一，通常需要填充或截断到统一长度
        # 这里假设统一到 10 秒 (160000 个采样点)
        f.create_dataset('sgram', data=np.array(all_sgram))
        f.create_dataset('wave', data=np.array(all_wave))
        f.create_dataset('labels', data=np.array(all_labels))

    print(f">>> 特征提取完成，保存在: {output_h5}")

# 使用示例
# extract_st_features("D:/datasets/dcase2025/fan/train", "train_stgram_data.h5")
if __name__ == "__main__":
    extract_st_features("D:/tools/Pycode/CapsnetForAD/data/ToyCar/train",
                        "D:/tools/Pycode/CapsnetForAD/workresult/train_stgram_data.h5")
    extract_st_features("D:/tools/Pycode/CapsnetForAD/data/ToyCar/test", "D:/tools/Pycode/CapsnetForAD/workresult/test_stgram_data.h5")