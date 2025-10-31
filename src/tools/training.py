import os
import torch
import torch.nn as nn
from mlflow.evaluation import evaluation
from pandas.core.dtypes import inference
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
from sklearn import metrics

from src.network import capsnet
from src import config as cfg

def train(tr_x, tr_y, val_x, val_y):
    """训练神经网络"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建模型
    model = capsnet.gccaps(input_shape=tr_x.shape[1:], n_classes=tr_y.shape[1])
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(tr_x), torch.FloatTensor(tr_y)),
        batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y)),
        batch_size=cfg.batch_size, shuffle=False
    )

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate['initial'])
    criterion = nn.BCELoss()

    # 训练配置
    n_epochs = cfg.n_epochs if cfg.n_epochs > 0 else 10000
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_eer': [], 'val_f1': []
    }

    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct = 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += ((outputs > 0.5).float() == batch_y).float().mean().item()

        # 验证阶段
        model.eval()
        val_loss, val_correct = 0, 0
        all_targets, all_predictions = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_correct += ((outputs > 0.5).float() == batch_y).float().mean().item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        # 计算指标
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_loader)

        # 计算EER和F1
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        eer = evaluation.compute_eer(all_targets.flatten(), all_predictions.flatten())
        f1 = metrics.f1_score(all_targets,
                              inference.binarize_predictions_2d(all_predictions, 0.5),
                              average='micro')

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_eer'].append(eer)
        history['val_f1'].append(f1)

        # 打印进度
        print(f'Epoch {epoch + 1}/{n_epochs}: '
              f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, '
              f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, '
              f'val_eer: {eer:.4f}, val_f1: {f1:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(cfg.model_path, f'gccaps_best.pth'))
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience and cfg.n_epochs == -1:
            print(f'早停于第 {epoch + 1} 轮')
            break

        # 学习率衰减
        if epoch % cfg.learning_rate['decay_rate'] == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.learning_rate['decay']

    # 保存最终模型和训练历史
    torch.save(model.state_dict(), os.path.join(cfg.model_path, 'gccaps_final.pth'))
    np.save(os.path.join(cfg.model_path, 'training_history.npy'), history)

    return model, history


# 如果需要TensorBoard，可以单独添加
def setup_tensorboard():
    """设置TensorBoard"""
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(cfg.log_path)
    return writer

# 在训练循环中添加TensorBoard记录
# writer.add_scalar('Loss/train', train_loss, epoch)
# writer.add_scalar('Accuracy/train', train_acc, epoch)
