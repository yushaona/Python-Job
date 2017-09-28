print("--------闰年计算器----------")
while 1:
    try:
        year = int(input("请输入年份"))
        if year % 3200 == 0 and year % 172800 == 0:
            print(str(year) + "是闰年")
        elif year % 400 == 0:
            print(str(year)+"是闰年")
        elif year % 4 == 0:
            if year % 100 != 0:
                print(str(year)+"是闰年")
            else:
                print(str(year)+"是平年")
        else:
            print(str(year) + "是平年")
    except ValueError:
        print("输入不合法,重新输入年份")
print("Good Bye!")