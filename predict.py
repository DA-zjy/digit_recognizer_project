# predict.py

import torch
import pandas as pd
from tqdm import tqdm

# 从我们自己的模块中导入
import config
from model import ResNet18
from dataset import get_test_loader # 我们只需要测试数据加载器

def predict():
    """使用最佳模型进行预测，并生成提交文件"""
    print("开始预测...")
    
    # 1. 获取测试数据加载器
    test_loader = get_test_loader()
    
    # 2. 实例化模型并移动到设备
    model = ResNet18().to(config.DEVICE)
    
    # 3. 加载训练好的最佳模型权重
    best_model_path = f"{config.CHECKPOINT_PATH}/digit_recognizer_epoch_8.pth"
    model.load_state_dict(torch.load(best_model_path))
    
    # 4. 切换到评估模式
    model.eval()
    
    # 5. 存储预测结果
    all_predictions = []
    
    # 6. 开始预测循环
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting"):
            images = images.to(config.DEVICE)
            
            # --- 使用和你evaluate函数中完全一样的TTA策略 ---
            # 原始图片预测
            outputs_original = model(images)
            probs_original = torch.softmax(outputs_original, dim=1)
            
            # 水平翻转图片预测
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            probs_flipped = torch.softmax(outputs_flipped, dim=1)
             
            # 融合预测概率
            avg_probs = (probs_original + probs_flipped) / 2
            
            # 基于融合概率获得最终预测
            _, predicted = torch.max(avg_probs, 1)
            # ----------------------------------------------------
             
            # 将一个批次的预测结果添加到总列表中
            all_predictions.extend(predicted.cpu().numpy())

    print("预测完成！")
    
    # 7. 生成提交文件
    print("正在生成提交文件...")
    submission = pd.DataFrame({
        "ImageId": list(range(1, len(all_predictions) + 1)),
        "Label": all_predictions
    })
    
    submission_path = "./submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已成功生成在: {submission_path}")

if __name__ == '__main__':
    predict()