PR-PL: A Novel Transfer Learning Framework with Prototypical Representation based Pairwise Learning for EEG-Based Emotion Recognition
=
* A Pytorch implementation of our paper "PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals." <br>
* 这篇提出了一种基于原型表示的成对学习（PR-PL）的新型迁移学习框架，以学习个体间情感揭示的判别性和广义原型表示，并将情感识别制定为成对学习，以减轻对精确标签信息的依赖。 更具体地说，开发了一种基于原型学习的对抗性判别域适应方法来编码脑电图数据固有的与情感相关的语义结构，同时开发了采用自适应伪标签方法的成对学习，以实现带有噪声标签的可靠且稳定的模型学习。 通过域适应，源域和目标域的特征表示在共享特征空间上对齐，同时还考虑了源域和目标域的特征可分离性。
* ![image](https://github.com/user-attachments/assets/a8a92423-00ba-4f51-8abc-d484a9f589da)


# 需要安装:
* Python 3.7
* Pytorch 1.3.1
* NVIDIA CUDA 9.2
* Numpy 1.20.3
* Scikit-learn 0.23.2
* scipy 1.3.1

# 准备
* 需要的数据集: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)

# Training 
需要安装Python 3.7；Pytorch 1.3.1；NVIDIA CUDA 9.2；Numpy 1.20.3；Scikit-learn 0.23.2；scipy 1.3.1

PR-PL 模型定义文件：model_PR_PL.py

PR-PL 的pipeline：implementation_PR_PL.py

领域对抗训练的实施：Adversarial.py

数据集准备：data_prepare_seed.m

修改设置（路径等）后，只需运行 implementation_PR_PL.py 中的 main 函数即可
# Acknowledgement
* The implementation code of domain adversarial training is bulit on the [dalib](https://dalib.readthedocs.io/en/latest/index.html) code base 
# Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.
@ARTICLE{10160130,
  author={Zhou, Rushuang and Zhang, Zhiguo and Fu, Hong and Zhang, Li and Li, Linling and Huang, Gan and Li, Fali and Yang, Xin and Dong, Yining and Zhang, Yuan-Ting and Liang, Zhen},
  journal={IEEE Transactions on Affective Computing}, 
  title={PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TAFFC.2023.3288118}}
