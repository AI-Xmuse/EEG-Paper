#-------------------------------------
# 主程序入口
# Date: 2024.4.24
# Author: Ming Jin
# All Rights Reserved
#-------------------------------------
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import torch
import datetime

# from chord import Chord

from torch.autograd import Variable
import torch.optim as optim
# import torch.optim.RAd
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter
import pytorch_warmup as warmup

# write_path = SummaryWriter('/home/ming/workspace/test_GAAT/GAAT/tensorboard')

from args import *
from model_PGCN import PGCN
from dataloader import load_data_de, load_data_inde
from utils import CE_Label_Smooth_Loss, set_logging_config, save_checkpoint
from node_location import convert_dis_m, get_ini_dis_m, return_coordinates

# from torchsummary import summary
np.set_printoptions(threshold=np.inf)

from thop import profile
from ptflops import get_model_complexity_info


class Trainer(object):

    def __init__(self, args, subject_name):
        self.args = args
        self.subject_name = subject_name

    def train(self, data_and_label):
        logger = logging.getLogger("train")   #创建了一个日志记录器，用于记录训练过程中的信息。
        laplacian_array = []  # 存放该subject优化后的laplacian matrix 列表
        train_set = TensorDataset((torch.from_numpy(data_and_label["x_tr"])).type(torch.FloatTensor),
                                  (torch.from_numpy(data_and_label["y_tr"])).type(torch.FloatTensor))
        val_set = TensorDataset((torch.from_numpy(data_and_label["x_ts"])).type(torch.FloatTensor),
                                (torch.from_numpy(data_and_label["y_ts"])).type(torch.FloatTensor))

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        # 局部视野的邻接矩阵
        adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9))).to(self.args.device)

        # 返回节点的绝对坐标
        coordinate_matrix = torch.FloatTensor(return_coordinates()).to(self.args.device)


        #####################################################################################
        # 2.define model
        #####################################################################################
        model = PGCN(self.args, adj_matrix, coordinate_matrix)

        lap_params, local_params, weight_params = [], [], []
        for pname, p in model.named_parameters():
        #     print(pname)
            #如果参数的名称是 "adj"，它被认为是拉普拉斯矩阵参数；如果参数名称中包含 "local"，它被认为是局部参数；否则，它被认为是普通的权重参数。
            if str(pname) == "adj":
                lap_params += [p]
            elif "local" in str(pname):
                local_params += [p]
            else :
                weight_params += [p]

        optimizer = optim.AdamW([
            {'params': lap_params, 'lr': self.args.beta},
            {'params': local_params, 'lr': self.args.lr},
            {'params': weight_params, 'lr': self.args.lr},
        ], betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        _loss = CE_Label_Smooth_Loss(classes=self.args.n_class, epsilon=self.args.epsilon).to(self.args.device)  #标签平滑交叉熵损失函数 CE_Label_Smooth_Loss。这个损失函数通常用于处理标签不完全确定的情况，通过添加一个小的值 epsilon 来平滑标签
        model = model.to(self.args.device)


        #############################################################################
        # 3.start train
        #############################################################################
        train_epoch = self.args.epochs

        # warm up
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[train_epoch // 3],
                                                            gamma=0.1)     #学习率会在训练了总轮数的三分之一后降低到原来的 10%
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)    #创建一个“热身”调度器，它会在训练的前几个步骤中逐渐增加学习率
        warmup_scheduler.last_step = -1     #设置“热身”调度器的最后步骤为 -1，这可能是用来标记“热身”阶段尚未开始

        best_val_acc = 0

        for epoch in range(train_epoch):
            epoch_start_time = time.time()
            train_acc = 0
            train_loss = 0
            val_loss = 0
            val_acc = 0

            model.train()
            for i, (x, y) in enumerate(train_loader):
                model.zero_grad()  # 清空上一步残余更新参数值

                x, y = x.to(self.args.device), y.to(device=self.args.device, dtype=torch.int64)
                output, lap_1, _ = model(x)
                loss = _loss(output, y)
                loss.backward()  # 误差反向传播，计算参数更新值

                optimizer.step()  # 将参数更新值施加到net的parmeters上

                if i < len(train_loader) - 1:
                    with warmup_scheduler.dampening():
                        pass

                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
                train_loss += loss.item() * y.size(0)
            #计算并更新训练准确率和损失的平均值。
            train_acc = train_acc / train_set.__len__()
            train_loss = train_loss / train_set.__len__()

            with warmup_scheduler.dampening(): #执行“热身”调度器的阻尼步骤，并更新学习率调度器
                lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for j, (a, b) in enumerate(val_loader):
                    a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                    output, lap, fused_feature = model(a)

                    val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())
                    batch_loss = _loss(output, b)
                    val_loss += batch_loss.item() * b.size(0)

            val_acc = round(float(val_acc / val_set.__len__()), 4)
            val_loss = round(float(val_loss / val_set.__len__()), 4)

            is_best_acc = 0
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                is_best_acc = 1

            if epoch == 0:
                logger.info(self.args)

            if epoch % 5 == 0:  #每过 5 个轮次，记录当前轮次的验证准确率和损失。
                logger.info("val acc and loss on epoch_{} are: {} and {}".format(epoch, val_acc, val_loss))

            save_checkpoint({
                'iteration': epoch,
                'enc_module_state_dict': model.state_dict(),
                'test_acc': val_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best_acc, self.args.log_dir, self.subject_name)   #保存模型和优化器的状态。save_checkpoint 函数可能接受当前轮次、模型状态、验证准确率、是否是最佳准确率、日志目录和受试者名称作为参数。

            if best_val_acc == 1:
                break
        # self.writer.close()
        return best_val_acc, laplacian_array




def main():
    args = parse_args()
    print("")
    print(f"Current device is {args.device}.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 将当前的实验结果存储到按秒组织的文件夹中
    datatime_path = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d-%H:%M:%S').replace(':', '-')
    args.log_dir = os.path.join(args.log_dir, args.dataset, datatime_path)
    set_logging_config(args.log_dir)
    logger = logging.getLogger("main")
    logger.info("Logs and checkpoints will be saved to：{}".format(args.log_dir))

    # summary(model, input_size=(62, 5))

    acc_list = []
    acc_dic = {}
    count = 0
    true_path = os.path.join(args.datapath, str(args.session))   #获取数据的路径
    for subject in os.listdir(true_path):  #，并遍历该路径下的所有受试者数据
        count += 1
        # print(subject)
        # load data of every single subject, 对于de和inde的设定，所读取的数据有所区别
        data_and_label = None
        subject_name = str(subject).strip('.npy')
        if args.mode == "dependent":
            logger.info(f"Dependent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_de(true_path, subject)
            # print(data_and_label)
        elif args.mode == "independent":
            logger.info(f"Independent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_inde(true_path, subject)
        else:
            raise ValueError("Wrong mode selected.")

        trainer = Trainer(args, subject_name)
        valAcc, lap_array = trainer.train(data_and_label)

        acc_list.append(valAcc)
        lap_array = np.array(lap_array)
        acc_dic[subject_name] = valAcc   #存储每个受试者的准确率
        logger.info("Current best acc is : {}".format(acc_dic))
        logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof=1)))


if __name__ == "__main__":
    main()