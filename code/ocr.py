import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import math
import random
import os
import json

"""
OCR神经网络类
功能：基于数据矩阵和标签进行手写数字识别
支持：初始训练、持续训练（调用train()）、预测识别（调用predict()）

权重可以保存到文件(NN_FILE_PATH)中，以便初始化时重新加载
"""
class OCRNeuralNetwork:
    LEARNING_RATE = 0.1      # 学习率，控制权重更新的步长
    WIDTH_IN_PIXELS = 20     # 输入图像的宽度（20x20像素）
    NN_FILE_PATH = 'nn.json' # 神经网络权重保存文件路径

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, training_indices, use_file=True):
        """
        初始化神经网络
        
        参数:
            num_hidden_nodes: 隐藏层节点数量
            data_matrix: 所有样本的特征数据
            data_labels: 所有样本的标签
            training_indices: 训练集索引列表
            use_file: 是否从文件加载已保存的权重
        """
        # 向量化激活函数及其导数，使其能处理矩阵运算
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        self._use_file = use_file
        self.data_matrix = data_matrix
        self.data_labels = data_labels

        # 如果没有保存的权重文件或不允许使用文件，则重新初始化权重
        if (not os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH) or not use_file):
            # 步骤1：用小随机数初始化权重（范围：-0.06 到 0.06）
            # theta1: 输入层到隐藏层的权重矩阵（维度：隐藏节点数 × 400）
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            # theta2: 隐藏层到输出层的权重矩阵（维度：10 × 隐藏节点数）
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            # 输入层偏置项
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            # 隐藏层偏置项
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

            # 使用训练集数据进行训练
            TrainData = namedtuple('TrainData', ['y0', 'label'])
            self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in training_indices])
            self.save()  # 保存训练好的权重
        else:
            self._load()  # 从文件加载权重

    def _rand_initialize_weights(self, size_in, size_out):
        """
        随机初始化权重矩阵
        返回维度为 (size_out, size_in) 的矩阵，值在 -0.06 到 0.06 之间
        """
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    def _sigmoid_scalar(self, z):
        """
        Sigmoid激活函数（标量版本）
        将输入值压缩到0-1之间，实现非线性变换
        """
        return 1 / (1 + math.e ** -z)

    def _sigmoid_prime_scalar(self, z):
        """
        Sigmoid函数的导数
        用于反向传播计算梯度
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def _draw(self, sample):
        """
        绘制图像样本（调试用）
        将400个像素点重新排列为20x20的图像并显示
        """
        pixelArray = [sample[j:j+self.WIDTH_IN_PIXELS] for j in range(0, len(sample), self.WIDTH_IN_PIXELS)]
        plt.imshow(list(zip(*pixelArray)), cmap = cm.Greys_r, interpolation="nearest")
        plt.show()

    def train(self, training_data_array):
        """
        使用反向传播算法训练神经网络
        
        训练步骤：
        1. 前向传播：计算每层的输出
        2. 计算输出误差
        3. 反向传播误差到隐藏层
        4. 更新权重和偏置
        """
        for data in training_data_array:
            # 兼容两种数据格式：字典或namedtuple
            if isinstance(data, dict):
                y0_data = data['y0']      # 输入特征（400维向量）
                label_data = data['label'] # 真实标签（0-9）
            else:
                y0_data = data.y0
                label_data = data.label
            
            # 步骤2：前向传播
            # 隐藏层：输入 × theta1 + 偏置，然后经过sigmoid激活
            y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(y0_data).T)
            sum1 = y1 + np.asmatrix(self.input_layer_bias)
            y1 = self.sigmoid(sum1)

            # 输出层：隐藏层输出 × theta2 + 偏置，然后经过sigmoid激活
            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)
            y2 = self.sigmoid(y2)

            # 步骤3：反向传播
            # 将标签转换为one-hot编码（如数字5 -> [0,0,0,0,0,1,0,0,0,0]）
            actual_vals = [0] * 10
            actual_vals[label_data] = 1
            output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)  # 输出层误差
            # 隐藏层误差 = (theta2转置 × 输出层误差) * sigmoid导数
            hidden_errors = np.multiply(
                np.dot(np.asmatrix(self.theta2).T, output_errors), 
                self.sigmoid_prime(sum1)
            )

            # 步骤4：更新权重（梯度下降法）
            # 更新theta2（隐藏层到输出层的权重）
            self.theta1 += self.LEARNING_RATE * np.dot(np.asmatrix(hidden_errors), np.asmatrix(y0_data))
            # 更新theta1（输入层到隐藏层的权重）
            self.theta2 += self.LEARNING_RATE * np.dot(np.asmatrix(output_errors), np.asmatrix(y1).T)
            # 更新偏置项
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        """
        对输入的图像数据进行预测
        
        参数:
            test: 400维的图像特征向量
        
        返回:
            预测的数字（0-9）
        """
        # 前向传播（与训练时相同，但不需要反向传播）
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(test).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)
        y2 = self.sigmoid(y2)

        # 返回输出值最大的神经元的索引（即预测的数字）
        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        """将神经网络的权重保存到JSON文件"""
        if not self._use_file:
            return

        json_neural_network = {
            "theta1":[np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2":[np_mat.tolist()[0] for np_mat in self.theta2],
            "b1":self.input_layer_bias[0].tolist()[0],
            "b2":self.hidden_layer_bias[0].tolist()[0]
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        """从JSON文件加载神经网络的权重"""
        if not self._use_file:
            return

        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        # 恢复权重和偏置
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]