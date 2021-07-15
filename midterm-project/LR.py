import pandas as pd
import numpy as np
from sklearn import linear_model
import csv
import matplotlib.pyplot as plt
if __name__=='__main__':
    lamda = 1  # 超参，为1可能是最优解
    df = pd.read_csv('train.csv')
    fd = pd.read_csv('test.csv')  # 测试集
    z = fd.iloc[:,1:385]  # 测试集
    x = df.iloc[:,1:385]
    y = df.iloc[:,385:386]
    X = np.mat(x)
    Y = np.mat(y)
    Z = np.mat(z)
    X_b = np.c_[np.ones((25000, 1)),x]
    Z_b = np.c_[np.ones((25000, 1)),z]
    theta = np.linalg.inv(((X_b.T).dot(X_b)) + lamda * np.eye(385)).dot(X_b.T).dot(Y)
    predict = (X_b).dot(theta)  # 训练集预测结果
    loss = sum(abs(predict - y)) / 25000
    print(loss)  # 训练集上损失，绝对值误差
    # predict_ = (Z_b).dot(theta)# 将测试集预测结果写入文件
    # with open("2.csv","w",newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in range(25000):
    #         writer.writerow([i,float(predict_[i])])
    # 标准库LR预测结果
    # model=linear_model.LinearRegression()
    # model.fit(X,Y)
    # print(model.intercept_)
    # print(model.coef_[0])
    # p=model.predict(Z)
    # with open("1.csv","w",newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in range(25000):
    #         writer.writerow([i,float(p[i])])


