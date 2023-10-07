#Task 1
"""
for var in range (1,11):
    print("\n Table for ",var,)
    for mult in range(1,11):
        print(var*mult,end=" ")
"""
#Task 2
"""
for i in range(0,20):
    print(2**i)
"""
#Task 3
"""
number=input("Enter a number: ")
sum = 0
for i in number:
    ber=int[i]
    if(ber %2 !=0):
        sum=sum+ber
print(sum)
"""
#Task 4
import random
def guess_game():
    num=random.int(1,10)
    for guess in range(1,6):
        guess=input()
        if guess==num:
            print("Congratulations,You have done it!")
            break
        else:
            print("Don't lose heart!!!")
print("The value of secret number is",num,"better luck next time")
guess_game()
