import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 以上是数据分析的三大件
import os
# os.sep根据你的操作系统平台自动选择适合的文件路径中的分隔符
path='data'+os.sep+'LogiReg_data.txt'
# pd.read_csv读取CSV（逗号分割）文件到DataFrame
#header:指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None
#name:用于指定列名，类型是（列表）（第一次考试成绩，第二次考试成绩，是否录取）
pdData=pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
# 使用head查看前几行数据（默认是前5行），不过你可以指定前几行
positive=pdData[pdData['Admitted']==1]
negative=pdData[pdData['Admitted']==0]
# 函数返回一个figure图像和一个子图ax的array列表。
fig,ax=plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=30,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=30,c='r',marker='x',label='No Admitted')
# 设置图标
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
def sigmoid(z):
    return 1/(1+np.exp(-z))
nums=np.arange(-10,10,step=1)
fig,ax=plt.subplots(figsize=(12,4))
ax.plot(nums,sigmoid(nums))
fig.show()
def model(X,theta):
    """
    :param X: 数据输入
    :param theta: 参数
    :return:
    """
    # np.dot() 矩阵相乘的意思 theta.T 转置
    return sigmoid(np.dot(X,theta.T))
pdData.insert(0,'Ones',1)
print(pdData)
orig_data=pdData.as_matrix()
print(orig_data)
cols=orig_data.shape[1]
print(cols)
X=orig_data[:,0:cols-1]
y=orig_data[:,cols-1:cols]
# 构造一行三列的theta参数
# 一行三列的参数
theta=np.zeros([1,3])
print(X.shape)
print(y.shape)
print(theta.shape)

def cost(X,y,theta):
    left=np.multiply(-y,np.log(model(X,theta)))
    right=np.multiply(1-y,np.log(1-model(X,theta)))
    return np.sum(left-right)/(len(X))
print(cost(X,y,theta))
def gradient(X,y,theta):
    # 先定义一个梯度 与参数的大小保持一致
    grad=np.zeros(theta.shape)
    print(grad[0,1])
    error=(model(X,theta)-y).ravel()
    print(error)
    for j in range(len(theta.ravel())): #for each parmeter
        term=np.multiply(error,X[:,j])
        # grad[0,j]表示第j个梯度
        grad[0,j]=np.sum(term)/len(X)
    return grad
print(gradient(X,y,theta))
#比较三种不同的梯度下降方法
#策略一：按照迭代次数进行停止  更新一次参数就试一次迭代
STOP_ITER=0
#查看迭代前后的目标函数 损失值没什么变化了 就停止迭代
STOP_COST=1
#根据梯度的变化 没啥变化了 就停止迭代
STOP_GRAD=2
def stopCriterion(type,value,threshold):
    #设定三种不同的停止策略
    if type==STOP_ITER:
        return value>threshold
    elif type==STOP_COST:
        return abs(value[-1]-value[-2])<threshold
    elif type==STOP_GRAD:
        return np.linalg.norm(value)<threshold
#当做迭代更新的时候 要把数据进行洗牌 使得模型的泛化能力更强 吧数据打乱
import numpy.random
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols=data.shape[1]
    X=data[:,0:cols-1]
    y=data[:,cols-1:]
    return X,y

import time
def descent(data,theta,batchSize,stopType,thresh,alpha):
    #梯度下降求解
    init_time=time.time()
    i=0 #迭代次数
    k=0 #batch
    X,y=shuffleData(data)
    grad=np.zeros(theta.shape)#计算的梯度
    costs=[cost(X,y,theta)]#损失值
    while True:
        grad=gradient(X[k:k+batchSize],y[k:k+batchSize],theta)
        k+=batchSize #取batch数量个数据
        if k>=100:#大于总数据
            k=0
            X,y=shuffleData(data)#重新洗牌
        theta=theta-alpha*grad #参数更新
        costs.append(cost(X,y,theta))#计算新的损失
        i+=1
        if stopType==STOP_ITER:
            value=i
        elif stopType==STOP_COST:
            value=costs
        elif stopType==STOP_GRAD:
            value=grad
        if stopCriterion(stopType,value,thresh):
            break
    return theta,i-1,costs,grad,time.time()-init_time
def runExpe(data,theta,batchSize,stopType,thresh,alpha):
    n=100
    theta,iter,costs,grad,dur=descent(data,theta,batchSize,stopType,thresh,alpha)
    name="Original" if (data[:,1]>2).sum()>1 else "Scaled"
    name+="data - learning rate:{} -".format(alpha)
    # 选择梯度下降的停止策略
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    # 画图与展示
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta
n=100
runExpe(orig_data,theta,n,STOP_ITER,thresh=5000,alpha=0.000001)



