# EmT
EmT 旨在在广义跨主题脑电图情绪分类和回归任务中表现出色。 在 EmT 中，EEG 信号被转换为时间图格式，使用时间图构建模块 (TGC) 创建一系列 EEG 特征图。 然后提出了一种新颖的残差多视图金字塔 GCN 模块（RMPG）来学习该系列中每个 EEG 特征图的动态图表示，并将每个图的学习表示融合为一个标记。 此外，我们设计了一个具有两种类型的令牌混合器的时间上下文变换器模块（TCT）来学习时间上下文信息。 最后，特定于任务的输出模块（TSO）生成所需的输出。 
EmT由四个主要部分组成：（1）时间图构建模块（TGC）、（2）RMPG、（3）TCT和（4）任务特定输出模块（TSO）。 在 TGC 中，EEG 信号被转换为时间图格式，即一系列 EEG 特征图，如图 2 所示。RMPG 学习该系列中每个 EEG 特征图的动态图表示，并将每个图的学习表示融合为一个 令牌。 TCT 通过专门设计的令牌混合器来学习时间上下文信息。 最后，TSO 模块将相应地生成分类和回归任务所需的输出。

![image](https://github.com/user-attachments/assets/f4b5f360-5fd9-468b-bc54-ac92e247486f)

EmT 模型定义文件EmT.py
准备好数据集后，运行对应的主文件，例如SEED数据集，运行main-SEED.py

# Cite
Please cite our paper if you use our code in your work:

```
@ARTICLE{ding2024emtnoveltransformergeneralized,
      title={{EmT}: A Novel Transformer for Generalized Cross-subject {EEG} Emotion Recognition}, 
      author={Yi Ding and Chengxuan Tong and Shuailei Zhang and Muyun Jiang and Yong Li and Kevin Lim Jun Liang and Cuntai Guan},
      year={2024},
      eprint={2406.18345},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.18345}}
```
