import torch
import pandas as pd
import os
import numpy as np
import librosa
from pathlib import Path
import tempfile
import torch.nn.functional as F
from torch import optim, nn
from wandb.old.summary import h5py

# 导入你的模型和特征提取器
from src.network.resAcapsnet import create_resnet_capsnet
from src.tools.extract import LogmelExtractor, extract_dataset, load_features
from src.tools.newmain import robust_data_preprocessing


def load_trained_model_with_threshold(checkpoint_path, device='cpu'):
    """加载训练好的模型和阈值"""
    print("加载模型和阈值...")

    try:
        # 方法1: 使用 weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"标准加载失败: {e}")
        # 方法2: 使用旧版加载方式
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e2:
            print(f"所有加载方法都失败: {e2}")
            raise

    input_shape = checkpoint.get('input_shape', (311, 64))
    print(f"模型输入形状: {input_shape}")

    # 创建模型
    from src.network.resAcapsnet import create_resnet_capsnet
    model = create_resnet_capsnet(input_shape=input_shape)

    # 加载状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 获取阈值
    threshold_95 = checkpoint.get('threshold_95', 0.1)
    train_stats = checkpoint.get('train_anomaly_stats', {})

    print(f"使用95%阈值: {threshold_95:.6f}")
    if train_stats:
        print(f"训练集统计 - 均值: {train_stats.get('mean', 0):.6f}, 标准差: {train_stats.get('std', 0):.6f}")

    return model, threshold_95


class DCASECSVGenerator:
    """DCASE Task 2 CSV文件生成器（使用训练时计算的阈值）"""

    def __init__(self, model, feature_extractor, threshold_95, device='cuda'):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold_95 = threshold_95  # 使用训练时计算的阈值
        self.device = device
        self.model.to(device)
        self.model.eval()
        print(f"模型已加载到设备: {device}")
        print(f"使用训练时95%阈值: {threshold_95:.6f}")

    def extract_features_batch(self, test_audio_dir, machine_type):
        """使用与训练时完全相同的批量特征提取方法"""
        # 获取测试文件
        test_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]
        test_files.sort()

        if not test_files:
            print(f"在 {test_audio_dir} 中没有找到wav文件")
            return None, []

        print(f"找到 {len(test_files)} 个测试文件")

        # 创建临时文件存储特征
        temp_output = tempfile.mktemp(suffix='.h5')

        try:
            # 使用与训练时相同的特征提取流程
            extract_dataset(
                dataset_path=test_audio_dir,
                file_names=test_files,
                extractor=self.feature_extractor,
                clip_duration=10,  # 使用与训练时相同的时长
                output_path=temp_output,
                recompute=True,
                n_transforms_iter=None,
                device=self.device
            )

            # 加载特征
            features = load_features(temp_output)
            features = self.robust_data_preprocessing(features)

            print(f"批量提取特征完成，形状: {features.shape}")
            return features, test_files

        except Exception as e:
            print(f"批量特征提取错误: {e}")
            return None, []
        finally:
            # 清理临时文件
            if os.path.exists(temp_output):
                os.remove(temp_output)

    def robust_data_preprocessing(self, features):
        """与训练时相同的预处理"""
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        median = np.median(features)
        mad = np.median(np.abs(features - median))
        if mad > 0:
            features = (features - median) / (mad + 1e-8)

        features = np.clip(features, -10, 10)
        return features

    def predict_anomaly_score_batch(self, features_batch):
        """批量预测异常分数 - 直接使用模型的compute_anomaly_score方法"""
        anomaly_scores = []

        with torch.no_grad():
            for i in range(len(features_batch)):
                features = features_batch[i]  # (n_frames, n_mels)

                # 转换为tensor并添加batch维度
                if isinstance(features, np.ndarray):
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, n_frames, n_mels)
                else:
                    features_tensor = features.unsqueeze(0)

                features_tensor = features_tensor.to(self.device)

                # 直接使用模型的compute_anomaly_score方法
                anomaly_score = self.model.compute_anomaly_score(features_tensor)

                # 处理不同类型的返回值
                if isinstance(anomaly_score, torch.Tensor):
                    if anomaly_score.dim() == 0:  # 标量tensor
                        anomaly_score = anomaly_score.item()
                    else:  # 多维tensor，取均值
                        anomaly_score = anomaly_score.mean().item()
                elif isinstance(anomaly_score, np.ndarray):
                    # numpy数组，取均值
                    anomaly_score = float(anomaly_score.mean())
                else:
                    # 其他类型，直接转换
                    anomaly_score = float(anomaly_score)

                anomaly_scores.append(anomaly_score)

                if i % 50 == 0:
                    print(f"预测进度: {i}/{len(features_batch)}, 当前分数: {anomaly_score:.6f}")

        return anomaly_scores


    def generate_dcase_submission(self, test_audio_dir, output_dir, machine_type):
        """为DCASE Task 2生成提交文件（使用训练时95%阈值）"""

        print(f"为 {machine_type} 处理目录: {test_audio_dir}")

        # 使用批量特征提取（与训练时相同）
        features_batch, test_files = self.extract_features_batch(test_audio_dir, machine_type)

        if features_batch is None:
            print(f"无法为 {machine_type} 提取特征")
            return None, None

        print(f"使用阈值: {self.threshold_95:.6f}")

        # 批量预测异常分数
        print("开始批量预测异常分数...")
        anomaly_scores = self.predict_anomaly_score_batch(features_batch)

        # 收集所有异常分数
        anomaly_data = []
        for i, (filename, score) in enumerate(zip(test_files, anomaly_scores)):
            anomaly_data.append({
                'filename': filename,
                'anomaly_score': score
            })

        # 生成异常分数文件
        anomaly_df = pd.DataFrame(anomaly_data)
        anomaly_output = os.path.join(output_dir, f'anomaly_score_{machine_type}_section_00_test.csv')
        anomaly_df.to_csv(anomaly_output, index=False)

        # 生成决策结果文件（使用训练时95%阈值）
        decision_results = []
        normal_count = 0
        anomaly_count = 0

        for item in anomaly_data:
            decision = 1 if item['anomaly_score'] > self.threshold_95 else 0
            if decision == 1:
                anomaly_count += 1
            else:
                normal_count += 1

            decision_results.append({
                'filename': item['filename'],
                'decision_result': decision
            })

        # 保存决策结果文件
        decision_df = pd.DataFrame(decision_results)
        decision_output = os.path.join(output_dir, f'decision_result_{machine_type}_section_00_test.csv')
        decision_df.to_csv(decision_output, index=False)

        print(f"为 {machine_type} 生成的文件:")
        print(f"  - {anomaly_output}")
        print(f"  - {decision_output}")
        print(f"  正常样本: {normal_count}, 异常样本: {anomaly_count}")
        print(f"  异常比例: {anomaly_count / len(anomaly_data) * 100:.2f}%")

        # 显示统计信息
        scores = anomaly_df['anomaly_score']
        print(f"异常分数统计 - {machine_type}:")
        print(f"  最小值: {scores.min():.6f}")
        print(f"  最大值: {scores.max():.6f}")
        print(f"  均值: {scores.mean():.6f}")
        print(f"  95%分位数: {np.percentile(scores, 95):.6f}")

        return anomaly_df, decision_df


def create_feature_extractor():
    """创建特征提取器 - 与训练时相同"""
    # 使用与训练时相同的参数
    extractor = LogmelExtractor(
        sample_rate=16000,
        n_window=1024,
        hop_length=512,
        n_mels=64
    )

    return extractor


def generate_all_machine_types():
    """为所有机器类型生成提交文件（使用训练时阈值）"""

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型检查点路径
    checkpoint_path = 'D:/tools/Pycode/CapsnetForAD/workresult/newResnet_capsnet_results/resnet_capsnet_model.pth'

    # 机器类型列表
    machine_types = [
        'AutoTrash', 'BandSealer', 'CoffeeGrinder', 'HomeCamera',
        'Polisher', 'ScrewFeeder', 'ToyPet', 'ToyRCCar'
    ]

    # 输出目录
    team_name = 'YourTeam'
    system_name = 'ResNetCapsNet'
    output_base_dir = f'./teams/{team_name}/{system_name}'
    os.makedirs(output_base_dir, exist_ok=True)

    try:
        # 加载模型和阈值
        print("加载模型和阈值...")
        model, threshold_95 = load_trained_model_with_threshold(checkpoint_path, device=device)

        # 创建特征提取器
        print("创建特征提取器...")
        feature_extractor = create_feature_extractor()

        # 创建CSV生成器（传入阈值）
        csv_generator = DCASECSVGenerator(model, feature_extractor, threshold_95, device)
        # 为每个机器类型生成文件
        for machine_type in machine_types:
            test_audio_dir = f'D:/tools/Pycode/CapsnetForAD/data/test/{machine_type}'  # 根据实际路径调整

            if os.path.exists(test_audio_dir):
                print(f"\n{'=' * 50}")
                print(f"处理机器类型: {machine_type}")
                print(f"{'=' * 50}")

                anomaly_df, decision_df = csv_generator.generate_dcase_submission(
                    test_audio_dir, output_base_dir, machine_type
                )
            else:
                print(f"警告: 测试目录不存在: {test_audio_dir}")

        print(f"\n{'=' * 50}")
        print("DCASE Task 2 提交文件生成完成！")
        print(f"输出目录: {output_base_dir}")
        print(f"为 {len(machine_types)} 种机器类型生成了文件")
        print(f"使用的95%阈值: {threshold_95:.6f}")
        print(f"{'=' * 50}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()




# if __name__ == "__main__":
#     generate_all_machine_types()


def test_loaded_model():
    """测试加载的已训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'D:/tools/Pycode/CapsnetForAD/workresult/newResnet_capsnet_results/resnet_capsnet_model.pth'

    try:
        # 加载模型
        model, threshold_95 = load_trained_model_with_threshold(checkpoint_path, device)

        print("=== 加载模型测试 ===")

        # 使用批次大小为2
        test_input = torch.randn(2, 311, 64).to(device)

        with torch.no_grad():
            model.eval()

            reconstructed, caps_features = model(test_input)

            print(f"输入范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
            print(f"胶囊特征范围: [{caps_features.min():.6f}, {caps_features.max():.6f}]")
            print(f"重建范围: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")

            mse = F.mse_loss(reconstructed, test_input)
            print(f"MSE: {mse.item():.6f}")

            # 计算异常分数
            scores = model.compute_anomaly_score(test_input)
            print(f"异常分数: {scores}")
            print(f"训练阈值: {threshold_95:.6f}")

            if mse < 0.5 and np.max(scores) < 1.0:
                print("✅ 加载的模型工作正常！")
            else:
                print("❌ 加载的模型仍然有问题")

    except Exception as e:
        print(f"加载模型失败: {e}")

if __name__ == "__main__":
    # 1. 先测试加载的模型
    test_loaded_model()