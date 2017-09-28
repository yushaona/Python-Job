"""k-means 算法实现的核心思想:
    1.随机选取k个质点
    2.将每个样本点对k个质点分别计算欧拉距离，选出距离最小的质点作为样本点聚类的索引值n
    3.重新选取质点--方法:从0到k-1个索引值遍历，计算聚类索引值相同的样本点的均值，作为当前索引新的质点数据
    4.重复步骤2，计算每个样本点到新k个质点的距离，并选出距离最小的质点作为样本聚类索引值n，
    5.一直到每个样本点的聚类的索引值不再发生变化,说明样本点聚类结束
"""
import  numpy as np
# 1.随机选取k个质点,（这里的随机不是真的随机，也就是根据样本点，进行有规则的随机，而非天马行空)
def randCent(dataSet,k):
    #矩阵的维数
  #  x = np.ndim()
    #样本点特征数据 np.shape返回（m,n） m表示行数 n表示列数  针对的二维矩阵
    n = np.shape(dataSet)[1]
    #质点矩阵
    centroids = np.mat(np.zeros((k,n)))

    #计算每个维度
    for j in range(n):
        minV = min(dataSet[:,j])
        #获取某个维度的最大值并计算最大值和最小值之间的范围
        # max括号内的运算是矩阵运算，所以max返回的也是矩阵，但max里面只包含一个值
        # 这时，外面套一个float，就是取出max返回的矩阵当中那个值
        rangeJ = float(max(dataSet[:,j])-minV) #矩阵减法运算
        #
        centroids[:,j] = minV + rangeJ * np.random.rand(k,1)
    return  centroids

# 2.根据质点和样本点聚类数据,并将聚类结果保存下来(保存的数据:1.聚类的分类索引 2.样本点到质点的距离 )
def clusterData(centroids,clusterResult,dataCount,dataSet,distOp,k):
    clusterChanged = False
    for i in range(dataCount):
        #取出一个样本点
        data = dataSet[i,:]
        #样本点和每一个质点比较,返回最小的质点索引和距离
        minDis ,minIndex = findMinCent(data,k,centroids,distOp)

        if clusterResult[i,0]  != minIndex:
            clusterChanged = True
        #记录最新的聚类结果
        clusterResult[i,:] = minIndex,minDis ** 2

    return clusterChanged

#对每个质点计算，求取最小值
def findMinCent(data,k,centroids,distOp):
    minDis = np.inf
    minIndex = -1
    #遍历所有质点
    for i in range(k):
        centroid = centroids[i,:]
        dist = distOp(data,centroid)
        if dist < minDis:
            minDis = dist
            minIndex = i

    return  minDis,minIndex


# 3.根据聚类结果重置k个质点
def resetCentroids(centroids,clusterResult,dataSet,k):
    for centIndex in range(k):
        # 如果clusterResult中对应样本聚类编号为centIndex则映射表中为1，否则为0
        clusterMap = clusterResult[:,0].A == centIndex
        # 根据映射表获取属于某个质点聚类的样本索引
        clusterPointIndexes = np.nonzero(clusterMap)[0] #clusterMap是一个数组,np.nonzero(clusterMap)返回数组索引值 [0] 只要索引值的行值
        # 获取聚类样本点
        clusterPoints = dataSet[clusterPointIndexes]#每行数据
        # 将质点重置为聚类样本的均值
        # 0标表示按列求均值
        centroids [centIndex,:] = np.mean(clusterPoints,axis=0)#对列数据求均值




def kmeans(dataSet,k,distOp,centOp):
    centroids = centOp(dataSet,k)

    dataCount = np.shape(dataSet)[0]#数据集行数

    clusterResult = np.mat(np.zeros((dataCount,2)))

    while 1:
        clusterChanged = clusterData(centroids,clusterResult,dataCount,dataSet,distOp,k)

        if not clusterChanged:
            break

        resetCentroids(centroids,clusterResult,dataSet,k)
    return  centroids,clusterResult


