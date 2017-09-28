from tools.io.tsv import TvsReader
from tools.math import  eulerDistance
from algorithm.kmeans import  kmeans,randCent
import  numpy as np

# start from here
dataReader = TvsReader()
allData = dataReader.loadDateSet('kmeansPoints.txt', attrType=float)
dataMatrix = np.mat(allData)

centroids, clusterResult = kmeans(dataSet=dataMatrix, k=5, distOp=eulerDistance, centOp=randCent)
print(centroids)
print(clusterResult)