f = open("fileData.txt","r+",encoding='utf-8')
print("文件名:",f.name)
# line = f.read(80)
# print("读取的字符串：%s" %(line))


# oneline = f.readline()
# print("读取一行:%s"%(oneline))
# twoline = f.readline()
# print("读取第二行:%s"%(twoline))
#
# thirdline = f.readline(3)
# print("读取第二行:%s"%(thirdline))


#
f.seek(8,0)
f.truncate()


f.close()