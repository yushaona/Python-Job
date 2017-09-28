age = int(input("please input your dog's age:"))
print("")
if age < 0:
    print("Are you kiding me?")
elif age == 1:
    print("It's equal to human being 14 years old!")
elif age == 2:
    print("It's equal to human being 22 years old!")
else:
    human = 22 + (age-2)*5
    print("It's equal to human being "+str(human)+" years old !")
input("<Enter> to Esc")