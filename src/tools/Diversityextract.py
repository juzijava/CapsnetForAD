import os
import glob
import torch
from src.tools.extract import extract_dataset, LogmelExtractor, load_features
from src.config.paths import DATA_CONFIG, DEVICE_CONFIG


def extract_multiple_datasets_optimized(dataset_dirs=None, output_file=None, duration=None):
    """
    同时提取多个数据集的音频特征

    Args:
        dataset_dirs: 数据集目录列表，如 ['/path/dataset1', '/path/dataset2']
        output_file: 输出文件路径
        duration: 音频时长
    """

    # 使用配置参数，如果传参则覆盖配置
    dataset_dirs = dataset_dirs or [DATA_CONFIG['audio_dir']]  # 默认单个目录
    output_file = output_file or DATA_CONFIG['output_file']
    duration = duration or DATA_CONFIG['feature_extraction']['duration']

    all_files = []
    dataset_labels = []  # 记录每个文件来自哪个数据集

    for dataset_idx, audio_dir in enumerate(dataset_dirs):
        print(f"搜索目录 {dataset_idx + 1}: {audio_dir}")

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

        # 转换为列表
        unique_files = list(set([path for path, name in audio_files]))

        # 添加到总文件列表
        all_files.extend(unique_files)
        dataset_labels.extend([dataset_idx] * len(unique_files))

        print(f"  数据集 {dataset_idx + 1} 找到 {len(unique_files)} 个音频文件")

    print(f"总共找到 {len(all_files)} 个音频文件")

    if len(all_files) == 0:
        print("错误: 没有找到任何音频文件！")
        return None, None

    # 显示文件统计信息
    for dataset_idx, audio_dir in enumerate(dataset_dirs):
        dataset_files = [f for i, f in enumerate(all_files) if dataset_labels[i] == dataset_idx]
        print(f"\n数据集 {dataset_idx + 1} ({audio_dir}) 统计:")
        print(f"  文件数量: {len(dataset_files)}")
        if dataset_files:
            file_size = os.path.getsize(dataset_files[0]) / 1024  # KB
            print(f"  示例文件: {os.path.basename(dataset_files[0])} ({file_size:.1f} KB)")

    # 创建特征提取器
    extractor_config = DATA_CONFIG['feature_extraction']
    extractor = LogmelExtractor(
        sample_rate=extractor_config['sample_rate'],
        n_window=extractor_config['n_window'],
        hop_length=extractor_config['hop_length'],
        n_mels=extractor_config['n_mels']
    )

    print(f"开始提取特征，预计时间帧数: {extractor.output_shape(duration)[0]}")

    # 提取所有文件的特征
    file_names = [os.path.basename(f) for f in all_files]

    extract_dataset(
        dataset_path=None,  # 设置为None，因为我们已经提供了完整路径
        file_names=all_files,  # 直接传递完整路径列表
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

    # 返回特征和数据集标签
    return features, dataset_labels


if __name__ == "__main__":
    # 使用示例：提取两个数据集
    dataset_paths = [
        "D:/tools/Pycode/CapsnetForAD/data/ToyTrain/train",
        "D:/tools/Pycode/CapsnetForAD/data/ToyCar/train"
    ]

    features, dataset_labels = extract_multiple_datasets_optimized(
        dataset_dirs=dataset_paths,
    )