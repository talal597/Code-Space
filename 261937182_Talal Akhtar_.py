
            ##Talal Akhtar Joyia, 261937182

            ##Dictionaries

car_info = {
   1: {"Car_number":1, "Car_name":'Toyota Soarer', "Daily_rent": 5000, "Liability_Insurance": 600, "Full_Insurance":700, "Availability": 5},
   2: {"Car_number":2, "Car_name":'Toyota JZX 100', "Daily_rent":3000, "Liability_Insurance": 400, "Full_Insurance": 500, "Availability": 5},
   3: {"Car_number":3, "Car_name":'Honda S2000',"Daily_rent": 7000, "Liability_Insurance":700,"Full_Insurance": 800, "Availability":5},
   4: {"Car_number":4, "Car_name":'GMC Denali', "Daily_rent": 9000, "Liability_Insurance":900, "Full_Insurance": 1000, "Availability": 5}
}

rental_history = []
            ##Functions

def display():                  #To display car information
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Car Number       |Car name       |Daily Cost      |Liability        |Full       |Availability" )
    print("-----------------------------------------------------------------------------------------------------------------")

    
    for car_id in car_info:                     #Loop to unpack dictionary
        car = car_info[car_id]
        print("{:<17}  {:<17}  {:<17} {:<17} {:<17} {:<17}".format(car["Car_number"], car["Car_name"], car["Daily_rent"], car["Liability_Insurance"], car["Full_Insurance"], car["Availability"]))


def rent():                 #Function to handle a rent request
    display()
    car_num = int(input("Enter car number: "))
    while car_num not in [1, 2, 3, 4]:
        print("Please choose a valid positive car number between 1 to 4.")
        car_num = int(input("Enter car num: "))
    num_days = int(input("How many days are you renting the car for? "))
    while num_days <= 0:
        print("Number of days must be a positive integer.")
        num_days = int(input("How many days?"))
    for i in car_info:
        if car_info[i]["Car_number"] == car_num:
            car = car_info[i]
    if car["Availability"] == 0:
        print("Sorry. This car is currently unavailable.")
        options()                                                   #Above lines take user input and validate it, if not valid, it takes the input again
    insur = input("Which type of insurance would you like? (L/F)")
    insur = insur.upper()
    while insur not in ["L", "l", "f", "F"]:
        print("Please type F or L to choose insurance type.")
        insur = input("Which insurance would you like?  ")
        insur = insur.upper()
    if insur == "F" or "f":
        insur_cost = car["Full_Insurance"]
    elif insur == "L" or "l":
        insur_cost = car["Liability_Insurance"]
                                                                    #Above code checks and unpacks the insurance type from the car_info dictionary according to input
    basecost = car["Daily_rent"] * num_days
    tax = basecost * 0.05
    totalcost = basecost + insur_cost + tax
    rental = (car["Car_name"], num_days, basecost, insur_cost, tax, totalcost)
    rental_history.append(rental)                                   #These lines calculate the rent and tax, and append a car's rental information in the rental history list.
    car["Availability"] -= 1
    print("-------------        TG ENTERPRISES     ------------------\n")
    print(car["Car_name"],"/nDays rented:", num_days, "\nCost:", basecost, "\nInsurance:", insur_cost, "\nTax:", tax, "\nTotal with 5% tax:", totalcost)
    options()                                                       #Above line prints the receipt
    
def returnCar():                                                    #Function to handle car return
    car_num = int(input("Enter .car number: "))
    while car_num not in [1, 2, 3, 4]:
        print("Please choose a valid positive car number between 1 to 4.")
        car_num = int(input("Enter car num: "))
    for i in car_info:
        if car_info[i]["Car_number"] == car_num:
            car = car_info[i]
    if car["Availability"] >= 5:
        print("Sorry. That car has not been rented out yet.")
        options()                                                   #Above lines take the input and validate it.
    else:
        car["Availability"] += 1
    print("Thank you for returning the", car["Car_name"])
    options()                                                       #Above line increments the car availability if it is below 5

def total():                                                        #Function to display totals
    if not rental_history:
        print("No rentals to display.")                             #These lines are for input validation
    else:
        totaltax = 0
        totalinsurance = 0
        totalscost = 0                                              #Setting base variables for all totals
        print("----TOTALS----")
        print("{:<15}  {:<15}  {:<15}  {:<15} {:<15} {:<15} {:<15}".format("Sr. no.", "Name", "Days Rented", "Base cost", "Insurance cost", "Tax", "Total cost"))
        i = 0
        for carname, days, bcost, incost, tax, totalcost in rental_history:
            print("{:<15}  {:<15}  {:<15}  {:<15} {:<15} {:<15} {:<15}".format(i+1, carname, days, bcost, incost, tax, totalcost))
            totaltax += tax
            totalinsurance += incost
            totalscost += totalcost
            i = i+1                                                 #For loop to unpack information about a stored rental in the rental history list, and to display it using string formatting
                                                                    #And also incrementing the values in total.
        print("Total tax:", totaltax, "\nTotal insurance:", totalinsurance, "\nTotal Cost:", totalscost)
        options()
import sys                   #Sys library to exit program
def options():                  #To display more options everytime a user performs action
        usrInp = input("Would you like to see more options? (y/n) ")
        if usrInp.upper() == "Y":
            main()
        elif usrInp.upper() == "N":
            sys.exit()          #Sys.exit() stops the python program
        else:
            print("Error. Please choose y or n")
        if usrInp=="n":
            print("Thank you for using Talal car rental service")
            options()

def main():                            ##MAIN
    print("- Welcome to TAJ enterprises -")
    print("Choose what to do!")
    print("[1]Rent a car")
    print("[2]Return a car")
    print("[3]View totals")
    print("[4]Exit")
                                        #Above code displays the main menu
    opt = int(input("Choose your option: "))
    while opt not in [1,2,3,4]:
        print("Please choose between 1-4,")         
        opt = int(input("Choose your option: "))
    if opt == 1:
        rent()
    elif opt == 2:
        returnCar()
    elif opt == 3:
        total()
    elif opt == 4:
        exit()                          #This function calls the other main functions for the program to run.
    
main()
