import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
import os

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# 设置数据路径
data_dir = './Data/'  # 你的数据路径

# 读取所有CSV文件
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
# 假设你已经有标签数组
labels_array = [0, 0, 0, 0, 0, 1, 0, 0,1,1,0,0,0,0,1,1,0,0,0,1]  # 20个标签，0表示非焦虑，1表示焦虑
# 加载EEG数据并使用给定的标签
def load_eeg_data(csv_files, data_dir, labels_array, num_samples=85000):
    eeg_data = []
    labels = labels_array  # 使用给定的标签数组
    for idx, file in enumerate(csv_files):
        try:
            # 读取数据时忽略第一行的列名，并确保数据为float类型
            subject_data = pd.read_csv(os.path.join(data_dir, file), header=0, dtype=float).values
            # print(f"读取文件: {file}, 数据形状: {subject_data.shape}")  # 输出调试信息

            # 只取前num_samples个数据点
            if subject_data.shape[0] >= num_samples:
                subject_data = subject_data[:num_samples, :]
                eeg_data.append(subject_data)
            else:
                print(f"Skipping {file} due to insufficient data: {subject_data.shape}")
                continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    eeg_data = np.array(eeg_data)
    labels = np.array(labels)  # 确保标签是numpy数组
    return eeg_data, labels

# 加载数据和标签
X_data, y_data = load_eeg_data(csv_files, data_dir, labels_array)


# 划分训练集和测试集
if len(X_data) >= 20:
    # 进行数据划分，确保18个训练样本，2个测试样本
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=18, test_size=2, random_state=42)
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
else:
    print(f"数据集样本数量不足，当前样本数量为：{len(X_data)}，无法进行10/2的划分。")


# 数据预处理和滤波函数
def bandpass_filter(eeg_data, lowcut=1, highcut=75, fs=200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, eeg_data, axis=1)
    return filtered_data


# 下采样到200Hz
def preprocess_eeg_data(eeg_data, fs=1000, target_fs=200):
    # 基线校准
    eeg_data -= np.mean(eeg_data, axis=1, keepdims=True)

    # 下采样
    downsample_factor = fs // target_fs
    eeg_data_downsampled = eeg_data[:, ::downsample_factor, :]  # 下采样

    # 带通滤波
    eeg_data_filtered = bandpass_filter(eeg_data_downsampled)

    return eeg_data_filtered


# 如果数据集足够，进行数据预处理
X_train = preprocess_eeg_data(X_train)
X_test = preprocess_eeg_data(X_test)

# 打印处理后的数据形状（如果有数据）
print(f"处理后的训练集形状: {X_train.shape}")
print(f"处理后的测试集形状: {X_test.shape}")

# LSTM模型
model = Sequential()

# 第一个LSTM层
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())  # 加入批归一化

# 第二个LSTM层
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# 全连接层
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))  # 防止过拟合

# 输出层
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])



# 训练LSTM模型
history = model.fit(X_train, y_train, epochs=10, batch_size=1)


# 获取训练过程中的accuracy
train_accuracy = history.history['accuracy']  # 训练准确率

# 获取训练的轮数
epochs = range(1, len(train_accuracy) + 1)

# 绘制训练准确率图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')

# 设置图表标签和标题
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy ')
plt.legend()

# 显示图表
plt.grid(True)
plt.show()
plt.savefig('training_accuracy.png')
# 模型评估
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
