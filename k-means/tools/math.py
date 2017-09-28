import numpy as np
#欧拉公式 计算两个坐标点的距离
def eulerDistance(va,vb):
    return  np.sqrt(np.sum(np.power(va-vb,2)))