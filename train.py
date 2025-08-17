# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm # 引入tqdm，一个强大的进度条工具

# 从我们自己的模块中导入
import config
from model import ResNet34
from dataset import get_train_val_loaders

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    """训练一个epoch"""
    model.train() # 设置为训练模式
    running_loss = 0.0
    # 使用tqdm来显示进度条
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, device, val_loader, criterion):
    """在验证集上评估模型（使用测试时增强）"""
    model.eval() # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # 原始图片预测
            outputs_original = model(images)
            probs_original = torch.softmax(outputs_original, dim=1)
            
            # 水平翻转图片预测
            images_flipped = torch.flip(images, dims=[3])  # 沿宽度维度翻转
            outputs_flipped = model(images_flipped)
            probs_flipped = torch.softmax(outputs_flipped, dim=1)
            
            # 融合预测概率（平均）
            avg_probs = (probs_original + probs_flipped) / 2
            
            # 计算损失（使用原始图片的输出）
            loss = criterion(outputs_original, labels)
            running_loss += loss.item() * images.size(0)
            
            # 基于融合概率获得最终预测
            _, predicted = torch.max(avg_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def main(): 
    """主函数"""
    # 确保模型保存目录存在
    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)
        
    # 获取数据加载器
    train_loader, val_loader = get_train_val_loaders()
    
    # 实例化模型并移动到设备
    model = ResNet34().to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # 学习率衰减调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    

    print("开始训练...")
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, config.DEVICE, train_loader, optimizer, criterion)
        val_loss, accuracy = evaluate(model, config.DEVICE, val_loader, criterion)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}%")
        
        # 保存模型
        torch.save(model.state_dict(), f"{config.CHECKPOINT_PATH}/digit_recognizer_epoch_{epoch+1}.pth")

    print("训练完成！")

# 这是一个Python脚本的入口点。
# 当你直接运行 `python train.py` 时，__name__ 的值就是 "__main__"，于是 main() 函数就会被调用。
# 如果这个文件被其他脚本作为模块导入，__name__ 就不是 "__main__"，main() 就不会被执行。
if __name__ == '__main__':
    main()