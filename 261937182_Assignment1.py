flight = [['PIA','10:00','4:00'],['Emirates','8:00','2:00']] #default flights
                                                             # they are later passed in dictionary for maintaining admin records (in file handlig)

###########################################################################################
PiaDict = {
    'Name':"PIA",
    'Arrival':'4:00',
    'Departure': '10:00'
}
EmiratesDict = {

    'Name':'Emirates',
    'Arrival':"2:00",
    'Departure':'8:00'
}

f = open('DictionaryFLights','w')
f.close()


def addDictionary(name,departure,arrival):
    new = dict(NAME = name, DEPARTURE = departure, ARRIVAL = arrival)               ##DICTIONARY HANDLING
    a =False
    f = open('DictionaryFlights','r') 
    for line in f:
        if str(new) in line:
            a = True
    if a == False:
        f.close()
        f = open('DictionaryFlights','a')
        f.write('\n')
        f.write(str(new))
        f.close()


Joiya1_easterEgg = flight[0]
Joiya2_easterEgg = flight[1]
addDictionary(Joiya1_easterEgg[0],Joiya1_easterEgg[1],Joiya1_easterEgg[2])
addDictionary(Joiya2_easterEgg[0],Joiya2_easterEgg[1],Joiya2_easterEgg[2])
#########################################################################################################


seatsEmirates = [['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ]]
seatsPIA = [['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ],['O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ]]



def displayseatsPIA():
    print('~~~~~~~~~~~~~~~PIA SEATS ~~~~~~~~~~~')
    print('           A    B    C    D    E    F    G    H    I')                  ##SEATINGS
    for i in range(len(seatsPIA)):
        print('Row ',i+1,'.',seatsPIA[i])

def displayseatsEmirates():
    print('~~~~~~~~~~~~~~~Emirates SEATS ~~~~~~~~~~~')
    print('           A    B    C    D    E    F    G    H    I')
    for i in range(len(seatsEmirates)):
        print('Row ',i+1,'.',seatsEmirates[i])
#############################################################################################################3
def abc_to_123(t):
    if t in 'aA':
        return 1
    elif t in 'bB':
        return 2 
    elif t in 'Cc':
        return 3
    elif t in 'Dd':                                         ##SEAT COLUMN SELECTOR
        return 4
    elif t in 'eE':
        return 5
    elif t in 'Ff':
        return 6
    elif t in 'gG':
        return 7
    elif t in 'hH':
        return 8
    elif t in 'iI':
        return 9
######################################################################################################################
def userinterface():
    print('------------------------')
    print('1.Show Flights\n2.Book Ticket\n3.Cancel Booking\n4.Back to Login')
    inp1 = input('Choose: ')
    
    if inp1 == '1':
        for i in flight:
            print('NAME: ',i[0])
            print('Departure: ',i[1])
            print('Arrival: ',i[2])
            print('==============================================')

        displayseatsPIA()
        print('--------------------------------------------------------------')  
        displayseatsEmirates()
        
        userinterface()

    elif inp1 == '2':
        print('~~~~~~~~~~~~~~~~BOOKING MENU~~~~~~~~~~~~~~~')
        print('1.PIA\n2.Emirates\n')
        inp2 = input('Choose?: ')
        if inp2 not in '12':
            print('Enter a valid input')
            userinterface()

        if inp2 == '1':
            displayseatsPIA()
        elif inp2 == '2':
            displayseatsEmirates()

        row = input('Enter Row(1,2,3...)?: ')
        if row not in '123456789':
            print('Incorrect Row')
            userinterface()
        
        column = input('Enter Seat Alphabet(a,b,c...: )?: ')
        if column not in 'abcdefghiABCDEFGHI':
            print('Invalid SeatNO')
            userinterface()
        
        if inp2 == '1':
            o = seatsPIA[((int(row))-1)]

            if o[abc_to_123(column)-1] == 'X':
                print('-------------SEAT ALREADY BOOKED------------')
                userinterface()

            o[abc_to_123(column)-1] = 'X'
            passanger_name = input('Enter Name of passanger: ')

            #################################################################################
            f = open('SEAT_BOOKINGS','a')
            f.write(passanger_name)
            f.write(' PIA AIRLINE ')                    ##FILE HANDLING SEAT RESERVATIONS
            f.write(  row)                                             #FOR PIA
            f.write(column)
            f.write('\n')
            f.close()

            ########################################################################################

            print('~~~~~~~~~FILE HANDLING DONE SUCCESSFULLY~~~~~~~~~~~~')
            print('~~~~~~~~~~~~~~~BOOKED SUCCESSFULLY~~~~~~~~~~~~')
            userinterface()

        elif inp2 == '2':
            o = seatsEmirates[((int(row))-1)]

            if o[abc_to_123(column)-1] == 'X':
                print('-------------SEAT ALREADY BOOKED------------')
                userinterface()

            o[abc_to_123(column)-1] = 'X'
            passanger_name = input('Enter Name of passanger: ')
            #######################################################################################
            f = open('SEAT_BOOKINGS','a')
            f.write(passanger_name)
            f.write(' Emirates AIRLINE ')               ##FILE HANDLING RESERVATIONS FOR EMIRATES
            f.write(row)
            f.write(column)
            f.write('\n')
            f.close()
            #########################################################################################

            print('~~~~~~~~~FILE HANDLING DONE SUCCESSFULLY~~~~~~~~~~~~')
            print('~~~~~~~~~~~~~~~BOOKED SUCCESSFULLY~~~~~~~~~~~~')
            userinterface()

    elif inp1 == '3':
        print('~~~~~~~~~~~~~~~ Cancel BOOKING~~~~~~~~~~~~~~')
        print('Available flights:\n1.PIA\n2.Emirates\n\n')
        inp2 = input('Choose?: ')

        if inp2 == '1':
            displayseatsPIA()
        elif inp2 == '2':
            displayseatsEmirates()

        row = input('Enter Row(1,2,3...)?: ')
        if row not in '123456789':
            print('Incorrect Row')
            userinterface()
        
        column = input('Enter Seat Alphabet(a,b,c...: )?: ')
        if column not in 'abcdefghiABCDEFGHI':
            print('Invalid SeatNO')
            userinterface()
        
        if inp2 == '1':
            o = seatsPIA[((int(row))-1)]

            if o[abc_to_123(column)-1] == 'X':
                o[abc_to_123(column)-1] = 'O'
            
                print('~~~~~~~~~~~~~~~CANCELLED SUCCESSFULLY~~~~~~~~~~~~')
                userinterface()
            else:
                print('SEAT ISNT BOOKED')
                userinterface()

        elif inp2 == '2':
            o = seatsEmirates[((int(row))-1)]

            if o[abc_to_123(column)-1] == 'X':
                o[abc_to_123(column)-1] = 'O'
                print('~~~~~~~~~~~~~~~CANCELLED SUCCESSFULLY~~~~~~~~~~~~')
                userinterface()
            else:
                print('SEAT ISNT BOOKED')
                userinterface()

    elif inp1 == '4':
        main()
   
def addflight():
    a = []
    name = input('Enter Flight name: ')
    departure = input("Enter departure time: ")
    arrival = input('Enter arrival time: ')

    a.append(name)
    a.append(departure)
    a.append(arrival)
    flight.append(a)

    addDictionary(name,departure,arrival)

    """"
    f = open('New Flights','w')
    f.write(name)
    f.write('  ')
    f.write(departure)                      #no need for file handling through list as whole dictionaries are being append in files
    f.write('  ')
    f.write(arrival)
    f.write('\n')
    f.close()
    """

    print('Flight Added Successfully')
    print('File handling done successfully')
    admininterface()

def removeflight():
    name = input('Enter Name of flight to remove: ')
    for i in flight:
        if i[0] == name:
            flight.remove(i)
    print('REMOVED SUCCESSFULLY')
    admininterface()

def modifyflight():
    a = 1
    for i in range(len(flight)):
        print(i+1,'.',flight[i])
    inp = input('Choose Flight to modify(1,2,3...)?: ')
    if inp in '1234567890':
        airplane = int(inp)-1
        q = flight[airplane]
        print('1.Name\n2.Departure\n3.Arrival')
        j = input('Choose(1,2,3)?: ')
        z = input('Enter new: ')
        
        if j == '1':
            q[0] = z
            
        elif j == '2':
            q[1] = z
            
        elif j == '3':
            q[2] = z
        else:
            print('Invalid Input')
            admininterface()
        
def admininterface():
    print('~~~~~~~~~~~~Welcome Admin~~~~~~~~~~~')
    print('1.View Flights\n2.Add a flight\n3.Remove a flight\n4.Modify a flight\n5.main\n\n')
    inp1 = input('Choose(1,2,3,4,5)?: ')

    if inp1 == '5':
        main()

    elif inp1 == '1':
        for i in flight:
            print('NAME: ',i[0])
            print('Departure: ',i[1])
            print('Arrival: ',i[2])
            print('==============================================')
        admininterface()

    elif inp1 == '2':
        addflight()
        admininterface()

    elif inp1 == '3':
        removeflight()
        admininterface()
    
    elif inp1 == '4':
        modifyflight()
        admininterface()
    else:
        print('Invalid Entry')
        admininterface()


def main():
    checkuser= ['user','user123']
    checkadmin = ['admin','admin123']
    print('<---------------------------- WELCOME ------------------------>')


    print(PiaDict)





    print('_________________________TALAL AKHTAR JOIYA____________________')
    print('Enter user & password: ')
    print('++++++++++ x( SECRET x( +++++++++\n:)username: user & password: user123\n:)username: admin & password: admin123\n+++++++++++++++++++++++++++\n')
    inp1 = input('Enter Username: ')
    inp2 = input('Enter Password: ')

    if inp1 == checkuser[0] and inp2 == checkuser[1]:
        userinterface()

    elif inp1 == checkadmin[0] and inp2 == checkadmin[1]:
        admininterface()
    else:
        print('Invalid user or password')
        main()

main()