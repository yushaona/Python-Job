def printInfo(arg1,*arg):
    "打印任何传入的参数"
    print(arg1)
    for var in arg:
        print("test",end = "-")
        print(var)


printInfo(10)
printInfo(20,30,40)


