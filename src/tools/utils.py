from collections import OrderedDict
import csv
import json
import operator
import pickle
import time
import types

import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

URBANSOUND_LABELS = []
LABELS = ['air_conditioner',
          'car_horn',
          'children_playing',
          'dog_bark',
          'drilling',
          'engine_idling',
          'gun_shot',
          'jackhammer',
          'siren',
          'street_music'
          ]
"""Ordered list of class labels."""

LABELS_DICT = {s: i for i, s in enumerate(LABELS)}
"""Dictionary to map labels to their integer values."""


def read_metadata(path, weakly_labeled=True, format_type='auto'):
    """Read from the specified metadata file.

    Args:
        path (str): Path to metadata file.
        weakly_labeled (bool): Whether the data is weakly-labeled.
        format_type (str): 'sed' for SED format, 'classification' for label format, 'auto' for auto-detect
    """
    y = OrderedDict()

    with open(path, 'r') as f:
        # 自动检测格式
        if format_type == 'auto':
            first_line = f.readline().strip()
            f.seek(0)  # 回到文件开头

            if '\t' in first_line and len(first_line.split('\t')) == 4:
                format_type = 'sed'
            else:
                format_type = 'classification'

        for row in csv.reader(f, delimiter='\t' if format_type == 'sed' else ','):
            if format_type == 'classification':
                # UrbanSound8K 分类格式: filename, label
                if len(row) < 2:
                    continue  # 跳过空行

                name = row[0]
                try:
                    label = int(row[1])  # UrbanSound8K 使用数字标签
                except ValueError:
                    continue  # 跳过标题行或其他非数据行

                if name not in y:
                    y[name] = []

                y[name].append(label)

            else:
                # 原有的 SED 格式处理
                if len(row) < 4:
                    continue

                name = 'Y' + row[0]
                onset = float(row[1])
                offset = float(row[2])
                label = row[3]

                if name not in y:
                    y[name] = []

                if weakly_labeled:
                    y[name].append(LABELS_DICT[label])
                else:
                    y[name].append((label, onset, offset))

    names = list(y.keys())
    target_values = list(y.values())

    # Convert target values to binary matrix
    if weakly_labeled:
        target_values = MultiLabelBinarizer().fit_transform(target_values)

    return names, target_values


def pad_truncate(x, length):
    """Pad or truncate a tensor to a specified length.

    Args:
        x (torch.Tensor or np.ndarray): Input tensor.
        length (int): Target length.

    Returns:
        torch.Tensor: The tensor padded or truncated to the specified length.
    """
    # Convert to tensor if numpy array
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    x_len = x.shape[0]

    if x_len > length:
        # Truncate
        x = x[:length]
    elif x_len < length:
        # Pad with zeros
        padding_shape = (length - x_len,) + x.shape[1:]
        padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=0)

    return x


class TorchStandardScaler:
    """PyTorch implementation of StandardScaler"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None

    def fit(self, x):
        """Compute mean and standard deviation for later scaling"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        self.mean_ = torch.mean(x, dim=0)
        self.var_ = torch.var(x, dim=0, unbiased=False)
        self.scale_ = torch.sqrt(self.var_ + 1e-8)
        return self

    def transform(self, x):
        """Standardize data using computed mean and standard deviation"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet")

        # Ensure mean and scale are on same device as x
        mean = self.mean_.to(x.device)
        scale = self.scale_.to(x.device)

        return (x - mean) / scale

    def fit_transform(self, x):
        """Fit to data, then transform it"""
        return self.fit(x).transform(x)


def compute_scaler(x):
    r"""Compute mean and standard deviation values for the given data.

    Args:
        x (torch.Tensor or np.ndarray): 3D array used to compute the parameters.

    Returns:
        TorchStandardScaler: Scaler used for later transformations.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    # Reshape to 2D: (n_samples * n_timesteps, n_features)
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])

    scaler = TorchStandardScaler()
    scaler.fit(x_2d)

    return scaler


def standardize(x, scaler):
    r"""Standardize data using the given scaler.

    Args:
        x (torch.Tensor or np.ndarray): 3D array to standardize.
        scaler (TorchStandardScaler): Scaler used for transformation.

    Returns:
        torch.Tensor: The standardized data.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    original_shape = x.shape
    # Reshape to 2D for standardization
    x_2d = x.reshape(-1, original_shape[-1])
    # Transform
    y_2d = scaler.transform(x_2d)
    # Reshape back to original shape
    return y_2d.reshape(original_shape)


def read_predictions(path):
    """Read classification predictions from the specified pickle file.

    Args:
        path (str): Path of pickle file.

    Returns:
        tuple: Tuple containing names and predictions as torch.Tensor.
    """
    with open(path, 'rb') as f:
        names, preds = pickle.load(f)

    # Convert predictions to tensor if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()

    return names, preds


def write_predictions(names, preds, output_path, write_csv=True):
    """Write classification predictions to a pickle file and,
    optionally, a CSV file.

    Args:
        names (list): Names of the predicted examples.
        preds (torch.Tensor or np.ndarray): 2D or 3D array of predictions.
        output_path (str): Output file path.
        write_csv (bool): Whether to also write to a CSV file.
    """
    # Convert tensor to numpy for storage
    if isinstance(preds, torch.Tensor):
        preds_numpy = preds.cpu().numpy()
    else:
        preds_numpy = preds

    with open(output_path, 'wb') as f:
        pickle.dump((names, preds_numpy), f)

    if write_csv:
        write_predictions_to_csv(names, preds_numpy, output_path[:-1] + 'csv')


def write_predictions_to_csv(names, preds, output_path):
    """Write classification predictions to a CSV file.

    Format of each entry is::

        fname<tab>p_1<tab>p_2<tab>...<tab>p_L

    Args:
        names (list): Names of the predicted examples.
        preds (torch.Tensor or np.ndarray): 2D or 3D array of predictions.
        output_path (str): Output file path.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if len(preds.shape) < 3:
        preds = np.expand_dims(preds, axis=-1)

    with open(output_path, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for i, name in enumerate(names):
            for pred in preds[i].T:
                writer.writerow([name] + pred.tolist())


def read_training_history(path, ordering=None):
    """Read training.py.py history from the specified CSV file.

    Args:
        path (str): Path to CSV file.
        ordering (str): Column name to order the entries with respect to
            or ``None`` if the entries should remain unordered.

    Returns:
        list: The training.py.py history.
    """
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            columns = next(reader)
            history = []

            # 安全地读取每一行
            for line in reader:
                try:
                    # 只转换数值数据，跳过空行或格式错误的行
                    if line:  # 确保行不为空
                        row_data = []
                        for item in line:
                            try:
                                row_data.append(float(item))
                            except ValueError:
                                # 如果无法转换为float，保持原样
                                row_data.append(item)
                        history.append(tuple(row_data))
                except Exception as e:
                    print(f"跳过格式错误的行: {line}, 错误: {e}")
                    continue

        # 如果历史记录为空，返回空列表
        if not history:
            print("训练历史文件为空")
            return []

        # Return unordered list if no ordering is given
        if ordering is None:
            return history

        # 安全地确定排序索引
        try:
            idx = columns.index(ordering)
            # 检查索引是否在有效范围内
            if idx >= len(history[0]):
                print(f"警告: 排序索引 {idx} 超出数据范围 (0-{len(history[0]) - 1})")
                return history

            reverse = ordering in ['acc', 'val_acc', 'val_map', 'val_f1_score']
            return sorted(history, key=operator.itemgetter(idx), reverse=reverse)

        except ValueError:
            print(f"排序列 '{ordering}' 不在CSV文件中")
            print(f"可用的列: {columns}")
            return history
        except IndexError as e:
            print(f"排序时发生索引错误: {e}")
            return history

    except FileNotFoundError:
        print(f"训练历史文件未找到: {path}")
        return []
    except Exception as e:
        print(f"读取训练历史文件时发生错误: {e}")
        return []


def timeit(callback, message):
    """Measure the time taken to execute the given callback.

    This function measures the amount of time it takes to execute the
    specified callback and prints a message afterwards regarding the
    time taken. The `message` parameter provides part of the message,
    e.g. if `message` is 'Executed', the printed message is 'Executed in
    1.234567 seconds'.

    Args:
        callback: Function to execute and time.
        message (str): Message to print after executing the callback.

    Returns:
        The return value of the callback.
    """
    # Record time prior to invoking callback
    onset = time.time()
    # Invoke callback function
    x = callback()

    print('%s in %f seconds' % (message, time.time() - onset))

    return x


def log_parameters(params, output_path):
    """Write the given parameters to a file in JSON format.

    Args:
        params (dict or module): Parameters to serialize. If `params` is
            a module, the relevant variables are serialized.
        output_path (str): Output file path.
    """
    if isinstance(params, types.ModuleType):
        params = {k: v for k, v in params.__dict__.items()
                  if not k.startswith('_')}
    elif not isinstance(params, dict):
        raise ValueError("'params' must be a dict or a module")

    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)


def to_tensor(data, device=None):
    """Convert data to PyTorch tensor with automatic device placement.

    Args:
        data: Input data (numpy array, list, or tensor)
        device: Target device (None for auto-detection)

    Returns:
        torch.Tensor: Tensor on specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).float()
    elif isinstance(data, list):
        tensor = torch.tensor(data, dtype=torch.float)
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    return tensor.to(device)


def to_numpy(tensor):
    """Convert PyTorch tensor to numpy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        np.ndarray: Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def save_model(model, path, optimizer=None, scheduler=None, epoch=None):
    """Save model checkpoint.

    Args:
        model: PyTorch model
        path: Save path
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None, scheduler=None):
    """Load model checkpoint.

    Args:
        model: PyTorch model
        path: Checkpoint path
        optimizer: Optimizer for resuming training.py.py
        scheduler: Scheduler for resuming training.py.py

    Returns:
        int: Epoch number
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"Model loaded from {path}, epoch: {epoch}")

    return epoch


# Example usage
if __name__ == "__main__":
    # Test the PyTorch utilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test pad_truncate
    test_tensor = torch.randn(100, 64)
    padded = pad_truncate(test_tensor, 150)
    truncated = pad_truncate(test_tensor, 50)
    print(f"Original shape: {test_tensor.shape}")
    print(f"Padded shape: {padded.shape}")
    print(f"Truncated shape: {truncated.shape}")

    # Test scaler
    test_data = torch.randn(1000, 10, 64)  # (batch, time, features)
    scaler = compute_scaler(test_data)
    normalized = standardize(test_data, scaler)
    print(f"Normalized data mean: {normalized.mean():.6f}, std: {normalized.std():.6f}")