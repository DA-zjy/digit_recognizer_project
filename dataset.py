# train_df = pd.read_csv("./data/train.csv")
# test_df = pd.read_csv("./data/test.csv")
# print(test_df.head())

import pandas as pd
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from config import BATCH_SIZE, TRAIN_CSV_PATH, TEST_CSV_PATH

class DigitDataset(Dataset):
    """有标签的数据集（用于训练/验证）"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        first_row = self.df.iloc[index]
        label_value = torch.tensor(first_row.iloc[0], dtype=torch.long)  # 第0列是标签
        pixel_values = first_row.iloc[1:].values  # 第1列开始是784个像素
        image_np_array = pixel_values.reshape(28, 28)
        image_array = image_np_array.astype('uint8')
        pil_image = Image.fromarray(image_array)
        if self.transform:
            pil_image = self.transform(pil_image)
        return pil_image, label_value

class TestDigitDataset(Dataset):
    """无标签的测试数据集（用于推理生成提交文件）"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        first_row = self.df.iloc[index]
        pixel_values = first_row.values  # 全部784列都是像素，没有标签列
        image_np_array = pixel_values.reshape(28, 28)
        image_array = image_np_array.astype('uint8')
        pil_image = Image.fromarray(image_array)
        if self.transform:
            pil_image = self.transform(pil_image)
        return pil_image  # 只返回图像，没有标签

def get_train_val_loaders():
    """返回训练和验证的数据加载器"""
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机旋转、平移、缩放
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = DigitDataset(train_df, transform=transform)
    n_total = len(full_dataset)
    n_val = max(1, int(0.1 * n_total))  # 10% 做验证
    n_train = n_total - n_val

    # 固定随机种子，保证可复现划分
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def get_test_loader():
    """返回测试数据加载器（用于生成提交文件）"""
    test_df = pd.read_csv(TEST_CSV_PATH)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = TestDigitDataset(test_df, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_loader

# 为了向后兼容，保留原函数名
def get_dataloaders():
    """返回训练和验证的数据加载器"""
    return get_train_val_loaders()