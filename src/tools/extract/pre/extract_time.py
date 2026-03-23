import librosa
import numpy as np
import h5py
import os
from tqdm import tqdm


def extract_fan_pro(audio_dir, output_file):
    sr, n_mels, n_fft, hop_length = 16000, 128, 2048, 512
    # 自动识别 labels
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
    labels = [1 if 'anomaly' in n.lower() else 0 for n in audio_files]

    features_list = []
    print(f">>> 提取 Fan 384维动态特征: {audio_dir}")

    for f in tqdm(audio_files):
        y, _ = librosa.load(f, sr=sr, duration=10)
        # 1. Log-Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)

        # 2. 差分与加速度 (捕捉抖动)
        delta1 = librosa.feature.delta(log_S)
        delta2 = librosa.feature.delta(log_S, order=2)

        # 3. 拼接并标准化 [T, 384]
        combined = np.concatenate([log_S, delta1, delta2], axis=0).T
        combined = (combined - np.mean(combined, axis=0)) / (np.std(combined, axis=0) + 1e-6)
        features_list.append(combined.astype(np.float32))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f_h5:
        f_h5.create_dataset('F_fan', data=np.array(features_list))
        f_h5.create_dataset('labels', data=np.array(labels).astype(np.int64))
    print(f">>> 特征已保存: {output_file}")


if __name__ == "__main__":
    # 修改为你本地的路径
    TRAIN_DIR = "/data/ToyCar/train"
    TEST_DIR = "/data/ToyCar/test"
    RESULT_PATH = "/workresult"
    extract_fan_pro(TRAIN_DIR, os.path.join(RESULT_PATH, "train_car_384.h5"))
    extract_fan_pro(TEST_DIR, os.path.join(RESULT_PATH, "test_car_384.h5"))