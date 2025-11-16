import os
import glob
import torch
from src.tools.extract import extract_dataset, LogmelExtractor, load_features


def extract_audio_features_optimized(audio_dir=None, output_file=None, duration=None,
                                     sample_rate=16000, n_window=1024, hop_length=512,
                                     n_mels=64, recompute=True):
    """
    提取音频特征的优化版本

    参数:
    audio_dir: 音频文件目录
    output_file: 输出特征文件路径
    duration: 音频剪辑时长(秒)
    sample_rate: 采样率
    n_window: 窗口大小
    hop_length: 跳数长度
    n_mels: Mel滤波器数量
    recompute: 是否重新计算特征
    """

    print(f"搜索目录: {audio_dir}")

    # 使用集合避免重复文件
    audio_files = set()
    extensions = ['.wav', '.mp3', '.flac', '.m4a']

    for ext in extensions:
        # 同时搜索小写和大写扩展名
        patterns = [
            os.path.join(audio_dir, f"*{ext}"),
            os.path.join(audio_dir, f"*{ext.upper()}")
        ]

        for pattern in patterns:
            found_files = glob.glob(pattern)
            for file_path in found_files:
                # 检查文件是否真实存在且不是空文件
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    # 使用小写文件名作为唯一标识
                    base_name = os.path.basename(file_path).lower()
                    audio_files.add((file_path, base_name))

    # 转换为列表并去重
    unique_files = list(set([path for path, name in audio_files]))
    file_names = [os.path.basename(f) for f in unique_files]

    print(f"找到 {len(file_names)} 个唯一的音频文件")

    if len(file_names) == 0:
        print("错误: 没有找到任何音频文件！")
        return None

    # 显示文件统计信息
    print("前5个文件:")
    for i, name in enumerate(file_names[:5]):
        full_path = os.path.join(audio_dir, name)
        file_size = os.path.getsize(full_path) / 1024  # KB
        print(f"  {i + 1}. {name} ({file_size:.1f} KB)")

    # 创建特征提取器
    extractor = LogmelExtractor(
        sample_rate=sample_rate,
        n_window=n_window,
        hop_length=hop_length,
        n_mels=n_mels
    )

    print(f"开始提取特征，预计时间帧数: {extractor.output_shape(duration)[0] if duration else '未知'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取特征
    extract_dataset(
        dataset_path=audio_dir,
        file_names=file_names,
        extractor=extractor,
        clip_duration=duration,
        output_path=output_file,
        recompute=recompute,
        n_transforms_iter=None,
        device=device
    )

    # 验证结果
    features = load_features(output_file)
    print(f"成功提取特征，形状: {features.shape}")
    print(f"特征统计 - 均值: {features.mean():.4f}, 标准差: {features.std():.4f}")
    print(f"特征范围 - 最小值: {features.min():.4f}, 最大值: {features.max():.4f}")

    return features


if __name__ == "__main__":
    # 示例用法 - 根据需要修改这些参数
    audio_directory = "D:/tools/Pycode/CapsnetForAD/data/test/ToyRCCar"# 修改为你的音频目录
    output_path = "D:/tools/Pycode/CapsnetForAD/workresult/testfeatures.h5"  # 修改为输出文件路径
    clip_duration = 10  # 音频时长(秒)，None表示使用完整音频

    features = extract_audio_features_optimized(
        audio_dir=audio_directory,
        output_file=output_path,
        duration=clip_duration
    )