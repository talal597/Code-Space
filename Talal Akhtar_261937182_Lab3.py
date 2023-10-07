#Task 1

def repeater():
      """This function repeats users input 20 times"""   
      print(userinput*20)
      
userinput=input("Please type something: ")
repeater()


#Task 2
def pre_defined() :
    """This funtion  takes a input from a user that seperates it with commas and counts specific letters in it"""
    print (userinput.split())
    print(userinput.count(user_input))
userinput=input("Write a sentence here: ")
user_input=input("Which word you want to be counted in the written sentence?: ")
pre_defined()

#Task 3
def fav_movie(userinput="Spider Man") :
      """This function tells about the users favourite movie"""
      print("My Favourite movie is: " + userinput)
      
userinput=input("Tell your favourite movie: ")
fav_movie(userinput)
fav_movie()
#Task 4 
def calculator():
    """This function acts like a calculator"""
    x=int(a)
    y=int(b)
    def addition():
        """ This function adds the given values"""
        print("The sum of given values is: ",x+y)
        return
    def subtraction():
        """ This function subtracts the given values"""
        print("The difference of given values is: ",x-y)
        return
    def multiplication():
        """This funtion multiplies the given values"""
        print("The multiplied result of the given values are: ",x*y)
        return
    def division():
        """This fucntion divides the given values"""
        print("The divided result of the given values are: ",x/y)
        return
    addition()
    subtraction()
    multiplication()
    division()
a=input("Please enter a number: ")
b=input("Please enter a number: ")
calculator()
    
      
