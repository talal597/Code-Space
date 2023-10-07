#Task 1
task1=['1','2','3','5','6','7','8']
def list():
    lngth = len(task1)
    for i in range(lngth):
        if(i%2==0):
            task1[i]=task1[i].replace(task1[i],"hello")
    print(task1)
list()

#Task 2
task2=[1,2,3,4,5,6,7]
opposite=task2[::-1]
print("Reversed List: ",opposite)
#Task 3
listt=[]
for i in range(1,21):
    listt.append(i)

print(listt)
#Task 4
import random
listt=[]
while (len(listt)<=10):
    var=random.randint(0,100)
    if var not in listt:
        listt.append(var)
    else:
        pass
print(listt)

#Task 5
listt=[2,4,6,8,10]
var=1
for i in listt:
    var=i*var
    print(var)
    

    
    
