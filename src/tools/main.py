import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.network.capsnet import create_gccaps
from src.tools.train import prepare_data, H5AudioDataset, train_model
from src.config.config import DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, PATH_CONFIG, DEVICE_CONFIG  # 导入配置


def main():
    # 设备设置
    device = torch.device(DEVICE_CONFIG['device'])
    print(f"使用设备: {device}")

    # 准备数据
    print("准备数据...")
    X_train, X_test, y_train, y_test = prepare_data(DATA_CONFIG['output_file'])

    # 创建数据集
    train_dataset = H5AudioDataset(DATA_CONFIG['output_file'], labels=y_train)
    test_dataset = H5AudioDataset(DATA_CONFIG['output_file'], labels=y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)

    # 创建模型
    print("创建模型...")
    model = create_gccaps(
        input_shape=MODEL_CONFIG['input_shape'],
        n_classes=MODEL_CONFIG['n_classes']
    )
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAIN_CONFIG['step_size'],
        gamma=TRAIN_CONFIG['gamma']
    )

    # 训练模型
    print("开始训练...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=TRAIN_CONFIG['num_epochs'],
        device=device
    )

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'input_shape': MODEL_CONFIG['input_shape'],
        'num_classes': MODEL_CONFIG['n_classes']
    }, PATH_CONFIG['model_save_path'])

    print(f"训练完成！模型已保存为 '{PATH_CONFIG['model_save_path']}'")

    return model, history


if __name__ == "__main__":
    model, history = main()