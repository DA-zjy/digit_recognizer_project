# Kaggle Digit Recognizer Project

## 1. 项目概述

Kaggle的"手写数字识别"竞赛项目，目标是使用深度学习模型（ResNet18）对MNIST数据集进行分类，读入csv格式并输出submission.csv。

## 2. 项目结构

```
├── checkpoints/  # 存放训练好的模型权重
├── data/         # 存放从Kaggle下载的数据集
├── .secrets/     # 存放Kaggle API凭证 (已被.gitignore忽略)
├── config.py     # 存储所有超参数和配置
├── dataset.py    # 定义数据加载和预处理的Dataset类
├── model.py      # 定义CNN模型架构 (ResNet18)
├── train.py      # 主训练脚本
├── predict.py    # 生成提交文件的脚本
├── submission.csv # 生成的提交文件示例 
└── README.md     # 项目说明文件
```

## 3. 安装与环境配置
建议使用 Conda 来管理环境。

1.  **克隆仓库**
    ```bash
    git clone [https://github.com/DA-zjy/digit_recognizer_project.git](https://github.com/DA-zjy/digit_recognizer_project.git)
    cd digit_recognizer_project
    ```

2.  **创建并激活 Conda 环境**
    ```bash
    # 我们创建一个新的、干净的环境，名叫 digit_env
    conda create -n digit_env python=3.9 -y
    conda activate digit_env
    ```

3. **下载数据集**
    # This script will use the API token in .secrets/ to download data
    ```
    export KAGGLE_CONFIG_DIR=./.secrets
    kaggle competitions download -c digit-recognizer -p ./data
    unzip ./data/digit-recognizer.zip -d ./data
    ```
    
4.  **安装依赖**

    * **方法一（推荐）：使用 `requirements.txt` 一键安装**
      ```bash
      pip install -r requirements.txt
      ```
      **注意**: `requirements.txt` 中包含了 PyTorch 的 CPU 版本或你本地的 CUDA 版本。如果你的设备（比如没有 NVIDIA 显卡）需要不同版本的 PyTorch，请参考方法二。

    * **方法二：手动安装核心依赖**
      如果方法一出现问题（尤其是 PyTorch 的 CUDA 版本不匹配时），请手动安装：
      1.  **安装 PyTorch**: 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，根据你的系统和 CUDA 版本，获取并运行最适合你的安装命令。
      2.  **安装其他库**:
          ```bash
          pip install tqdm Pillow
          ```

## 4. 使用方法

### 训练模型
直接从命令行运行 `train.py` 脚本即可开始训练。
```bash
python train.py
```
* 所有超参数（如 Epochs, Batch Size, Learning Rate）都可以在 `config.py` 文件中进行修改。
* 训练过程中，模型权重将被自动保存到 `checkpoints/` 目录下。

### 进行预测
(如果已创建 `predict.py`)
1.  确保 `checkpoints/` 目录下已有训练好的模型文件（如 `best_model.pth`）。
2.  运行脚本：
    ```bash
    python predict.py
    ```