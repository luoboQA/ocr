"""
目标：确定隐藏层应该使用多少个隐藏节点
方法：将数据集分割为训练集和测试集，创建不同隐藏节点数量（5, 10, 15, ... 45）的网络
      测试每个网络的性能表现

最佳性能的节点数量将用于实际系统。
如果多个节点数量表现相似，选择最小的节点数以获得更小的网络和更少的计算量。
"""

import numpy as np
from ocr import OCRNeuralNetwork
from sklearn.model_selection import train_test_split  # 用于数据集的随机分割

def test(data_matrix, data_labels, test_indices, nn):
    """
    测试神经网络的识别准确率
    
    参数:
        data_matrix: 所有样本的特征数据矩阵
        data_labels: 所有样本的真实标签
        test_indices: 测试集样本的索引列表
        nn: 待测试的神经网络对象
    
    返回:
        平均识别准确率 (0-1之间的浮点数)
    """
    avg_sum = 0
    # 进行100次测试取平均值，减少随机性影响
    for j in range(100):
        correct_guess_count = 0
        # 对每个测试样本进行预测
        for i in test_indices:
            test = data_matrix[i]        # 获取测试样本的特征
            prediction = nn.predict(test) # 神经网络预测
            if data_labels[i] == prediction:  # 判断预测是否正确
                correct_guess_count += 1

        # 计算本次测试的准确率
        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100  # 返回平均准确率


# 加载数据样本和标签
# data.csv: 每行是一个20x20=400像素的图像数据（0表示黑，1表示白）
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter = ',').tolist()
# dataLabels.csv: 每行是对应图像的真实数字（0-9）
data_labels = np.loadtxt(open('dataLabels.csv', 'rb')).tolist()

# 创建训练集和测试集（5000个样本随机分割）
train_indices, test_indices = train_test_split(list(range(5000)))

print("PERFORMANCE")
print("-----------")

# 尝试不同数量的隐藏节点（5, 10, 15, ..., 45）
for i in range(5, 50, 5):
    # 创建具有i个隐藏节点的神经网络
    # 参数：隐藏节点数, 数据矩阵, 标签, 训练索引, 是否使用保存的权重文件
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print("{i} Hidden Nodes: {val}".format(i=i, val=performance))