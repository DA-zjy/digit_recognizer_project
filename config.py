# config.py

import torch

# --- 训练配置 ---
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 256 #Batch如何选择？
LEARNING_RATE = 0.1 # 0.1

# --- 模型保存配置 ---
CHECKPOINT_PATH = "./checkpoints"

#数据集路径
TRAIN_CSV_PATH = "./data/train.csv"
TEST_CSV_PATH = "./data/test.csv"