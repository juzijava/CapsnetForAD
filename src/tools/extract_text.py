import os
import glob
import torch
from src.tools.extract import extract_dataset, LogmelExtractor, load_features
from src.config.config import DATA_CONFIG, DEVICE_CONFIG  # 导入配置


def extract_audio_features_optimized(audio_dir=None, output_file=None, duration=None):
    """优化版的特征提取，避免重复文件"""

    # 使用配置参数，如果传参则覆盖配置
    audio_dir = audio_dir or DATA_CONFIG['audio_dir']
    output_file = output_file or DATA_CONFIG['output_file']
    duration = duration or DATA_CONFIG['feature_extraction']['duration']

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

    # 创建特征提取器（使用配置参数）
    extractor_config = DATA_CONFIG['feature_extraction']
    extractor = LogmelExtractor(
        sample_rate=extractor_config['sample_rate'],
        n_window=extractor_config['n_window'],
        hop_length=extractor_config['hop_length'],
        n_mels=extractor_config['n_mels']
    )

    print(f"开始提取特征，预计时间帧数: {extractor.output_shape(duration)[0]}")

    # 提取特征
    extract_dataset(
        dataset_path=audio_dir,
        file_names=file_names,
        extractor=extractor,
        clip_duration=duration,
        output_path=output_file,
        recompute=extractor_config['recompute'],
        n_transforms_iter=None,
        device=DEVICE_CONFIG['device']
    )

    # 验证结果
    features = load_features(output_file)
    print(f"成功提取特征，形状: {features.shape}")
    print(f"特征统计 - 均值: {features.mean():.4f}, 标准差: {features.std():.4f}")
    print(f"特征范围 - 最小值: {features.min():.4f}, 最大值: {features.max():.4f}")

    return features


if __name__ == "__main__":
    # 直接使用配置文件中的参数
    features = extract_audio_features_optimized()