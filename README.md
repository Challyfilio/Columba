# Aircraft Classification

---
使用ResNet进行迁移学习

## 目录结构及文件说明
```
./
├─dataset
│   ├─aircraft
│   │    ├─train
│   │    └─test
│   └─aircraft_similar
│        ├─train
│        └─test
├─output                # 模型输出，包含：保存的模型，日志，可视化图片，loss_acc折线图
├─pretrain_model        # 存放ResNet的预训练模型（ResNet18，ResNet50）
├─scripts               # 训练，测试脚本
│   ├─train.sh
│   └─eval.sh
├─utils                 # 工具类：日志，画图，训练
│   ├─__init__.py
│   ├─logger.py
│   ├─tools.py
│   └─train.py
├─main.py
└─trainer.py 
```

## How to Run
运行脚本在`scripts/`目录里，在`scripts/`目录下运行
```
bash train.sh [NET] [DS] [OTL] [EPOCH]
bash eval.sh [NET] [DS] [OTL] [EPOCH]
```
- NET   # 主干网络 [resnet50 , resnet18]
- DS    # 数据集 [aircraft , aircraft_similar]
- OTL   # 是否只训练最后的线性层 [0:False , 1:True]
- EPOCH # epochs

Example:
```
bash train.sh resnet50 aircraft 0 20
```

## 相关文件下载
- 数据集
  - [数据集](https://pan.baidu.com/s/1NxEcynlSSs4VAImesmXMpg) 提取码：3721
- output 训练结果
  - [output](https://pan.baidu.com/s/1ZzMoJtdEcHghRXevhQEhdg) 提取码：3721
- ResNet预训练模型
  - [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth)
  - [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth)
  - [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)
  - [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
  - [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)
