# Task 1
def print_left_right(one , two ):
    """Prints strings with spaces on 50 charector line """
    spaces = 50 - len(one) - len(two)
    # print(spaces)
    spc = " "*spaces
    print("{:<}{}{:>}".format(one,spc,two))
string1 = input("enter the first string: ")
string2 = input("enter the second string: ")
print(print_left_right(string1,string2))


##Task # 2


user = input("Enter a single chatrector")

if len(user) == 1 :
    var = user.lower()
    if var == "a":
        print("Vowel")
    if var == "e":
        print("Vowel")
    if var == "i":
        print("Vowel")
    if var == "o":
        print("Vowel")
    if var == "u":
        print("Vowel")
else :
    print("Error:Charector is not one alphabet")

##task 3 
# Function to convert integer to Roman values
def printRoman(number):
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
      
    while number:
        div = number // num[i]
        number %= num[i] 
        while div:
            print(sym[i], end = "")
            div -= 1
        i -= 1
number = int(input("Enter a i nteger you want to vonvert into roman: "))
print("Roman value is:", end = " ")
printRoman(number)
print()

##task 4

def Reverse_liner():
      ofile=open("testfile.txt","r")
      k=ofile.readlines()
##      print(k)
      t=reversed(k)
      for i in t:
           file =  open("outputfile.txt","a")
           file.write(i.lstrip())

Reverse_liner()

##task 5
def counter(fname): 
    # variable to store total word count
    num_words = 0     
    # variable to store total line count
    num_lines = 0     
    # variable to store total character count
    num_charc = 0     
    # variable to store total space count
    num_spaces = 0
    with open(fname, 'r') as f:
        for line in f:
            num_lines += 1
            word = 'Y'
            for letter in line: 
                if (letter != ' ' and word == 'Y'):
                    num_words += 1
                    word = 'N'
                elif (letter == ' '):
                    num_spaces += 1
                    word = 'Y'
                for i in letter:
                    if(i !=" " and i !="\n"):
                        num_charc += 1                      
    print("Number of words in text file: ",
          num_words)
    print("Number of lines in text file: ",
          num_lines)
    print('Number of characters in text file: ',
          num_charc)
    print('Number of spaces in text file: ',
          num_spaces)
fname = 'testfile.txt'
try:
    counter(fname)
except:
    print('File not found')
