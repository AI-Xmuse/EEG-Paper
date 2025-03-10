# PGCN: Pyramidal Graph Convolutional Network for EEG Emotion Recognition

![image](https://github.com/Jinminbox/PGCN/assets/48828942/662028f6-c272-4dd7-9b72-f0f65ca0c484)


![image](https://github.com/Jinminbox/PGCN/assets/48828942/26a5d2e1-3a76-4de8-9068-36d6bbaf026d)


这篇文章提出了金字塔图卷积网络（PGCN），它聚合了三个级别的特征：局部、介观和全局。 首先，构建了一个基于电极3D拓扑关系的普通GCN，用于集成二阶局部特征； 其次，基于先验知识构建了多个细观脑区域，并利用细观注意力顺序计算虚拟细观中心，以关注细观脑区域的功能连接；最后，融合节点特征及其3D位置来构建数值关系邻接矩阵，以从全局角度整合结构和功能连接。	模型总体架构如图2所示，可分为以下步骤。 (1)考虑到大脑网络频繁的局部连接，我们基于电极之间的空间距离构建稀疏结构邻接矩阵，并引入GCN来聚合局部特征。 （2）为了更好地区分不同脑区与情绪的相关性，我们基于先验研究构建了介观尺度的脑区，并计算了每个脑区的虚拟介观中心来表征介观特征。 （3）为了平衡长距离连接的重要性和经济性，我们将原始节点与虚拟介观节点融合，借助注意力机制构建稀疏全局图连接网络，并通过图卷积聚合全局特征。 (4)最后，将融合后的特征输入到3层全连接网络中以完成最终的情感识别任务。

1.安装虚拟环境，推荐 conda

2.获取数据后，通过处理 SEED4_pretrain.py 重新保存数据。检查 Raw Data Path 和 Resaved Data Path

3.运行主文件main_PGCN.py

```
@article{jin2024pgcn,
  title={PGCN: Pyramidal graph convolutional network for EEG emotion recognition},
  author={Jin, Ming and Du, Changde and He, Huiguang and Cai, Ting and Li, Jinpeng},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```
