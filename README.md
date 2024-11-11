# Jane Street Market Prediction
    
## 项目说明
本项目是 [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction) 比赛的解决方案。

- `notebooks/*.ipynb`: Jupyter notebooks用于数据探索、特征工程和模型原型验证
- `src/*.py`: 对应的Python脚本，用于模型训练和生产部署

## 环境准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据准备
1. 从[Kaggle比赛页面](https://www.kaggle.com/c/jane-street-market-prediction/data)下载数据
2. 解压数据到 `./data` 目录
```bash
mkdir -p data
cd data
unzip jane-street-market-prediction.zip
```

### 3. 创建模型目录
```bash
mkdir -p model
```

## 模型训练

### 后台训练
```bash
nohup python src/basic-pytorch-lstm.py > training.log 2>&1 &
```

## 模型说明
项目实现了两种模型：
1. LSTM
2. Transformer

可通过修改 `run_type` 参数选择不同模型：
```python
run_type = 'LSTM'  # or 'Transformer'
```

## 训练监控
使用wandb进行训练监控，首次运行需要：
```bash
wandb login
```

## 目录说明
- `data/`: 存放比赛数据
- `model/`: 存放训练好的模型
- `notebooks/`: 存放实验性代码
- `src/`: 存放生产训练代码

## 注意事项
1. 确保数据目录结构正确
2. 训练前检查GPU内存
3. 建议使用nohup运行长时间训练
4. 定期检查training.log和wandb面板

## License
MIT
