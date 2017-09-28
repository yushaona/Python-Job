# TsvRead class read data from the filename
class TvsReader(object):
    def __init__(self):
        pass
    def loadDateSet(self,filename,**kwargs):
        attrType = kwargs.get('attrType',None) # 这个类型主要指数据元素类型(int/float等)
        dataFile = open(filename,'r',encoding='utf-8')
        lines = dataFile.readlines()
        #解析每一行,生成行数组,matrix 最后形成多维数组
        matrix = [self.parseLine(line,attrType=attrType) for line in lines]
        return  matrix

    #每次返回一个数组
    def parseLine(self,line,**kwargs):
        attrType = kwargs.get('attrType',None)
        #拆分每一行数据
        words = line.strip().split('\t')
        if attrType is None:
            return  words
        #将每个元素类型转化以后返回
        return [ attrType(word) for word in words]
