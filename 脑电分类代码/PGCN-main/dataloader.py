import os
import numpy as np


def load_data_de(path, subject):#这个函数用于加载单个主题（subject）的数据

    dict_load = np.load(os.path.join(path, (str(subject))), allow_pickle=True)
    data = dict_load[()]['sample']
    label = dict_load[()]['label']
    split_index = dict_load[()]["clip"]   #从加载的字典中提取数据（'sample'）、标签（'label'）和分割索引（"clip"）

    x_tr = data[:split_index]
    x_ts = data[split_index:]
    y_tr = label[:split_index]
    y_ts = label[split_index:]     #根据分割索引将数据和标签分为训练集（x_tr, y_tr）和测试集（x_ts, y_ts）。

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }      #将分割后的数据和标签存储在一个字典data_and_label中，并返回这个字典

    return data_and_label


def load_data_inde(path, subject):   #这个函数用于加载多个主题的数据，并将其中一个主题的数据作为测试集，其余主题的数据合并为训练集。
    x_tr = np.array([])
    y_tr = np.array([])
    x_ts = np.array([])
    y_ts = np.array([])
    for i_subject in os.listdir(path):
        dict_load = np.load(os.path.join(path, (str(i_subject))), allow_pickle=True)  #对于每个文件，使用np.load加载数据
        # print(dict_load)
        data = dict_load[()]['sample']
        label = dict_load[()]['label']
        # print(data)
        # print(label)
        #如果当前文件的主题与subject参数匹配，则将这些数据和标签赋值给测试集变量（x_ts, y_ts）。如果不匹配，则将数据和标签添加到训练集变量中。
        if i_subject == subject:
            x_ts = data
            y_ts = label
        else:
            if x_tr.shape[0] == 0:
                x_tr = data
                y_tr = label
            else:
                x_tr = np.append(x_tr, data, axis=0)
                y_tr = np.append(y_tr, label, axis=0)

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }
    # print(data_and_label)
    return data_and_label

