print("-----------质数判断器------------")

while 1:
    try:
        num = int(input("请输入一个正整数"))
        if num < 1:
            print("请输入一个正整数")
            continue
        if num == 1:
            print("1既非质数,也非合数")
            continue
        for x in range(2,num):
            if num % x == 0:
                print(str(num)+"是合数")
                break
        else:
            print(str(num)+"是质数")

    except ValueError:
        print("输入数据类型错误,重新输入")