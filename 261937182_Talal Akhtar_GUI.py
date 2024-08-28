import tkinter as tk
from tkinter import END, Button, Entry, Label, filedialog, Text
import os
from tkinter import *
from tkinter import ttk
import tkinter
import csv
from tkinter import filedialog

root = tk.Tk()
root.geometry("250x500")
root.configure(bg='light blue')

l = Label( text = "BDL Enterprise")
l.config(font =("Times", 20))
def title(l):
    l.configure(bg="light blue")
    l.config(foreground="green")
    l.pack()
# using place method we can set the position of label
    l.place(relx = 0.5, rely = 0.1, anchor ='center')

title(l)
cartd = Frame()

cars = ttk.Treeview(cartd) 


cars['columns'] = ('No.', 'Model', 'Availibility', 'Price/Day', 'Liability Insurence/Day', 'Comprehensive Insurence/Day')

cars.column("#0", width=0,  stretch=NO)
cars.column('No.', anchor=W, width=30)
cars.column("Model",anchor=W, width=120)
cars.column("Availibility",anchor=W,width=80)
cars.column("Price/Day",anchor=W,width=80)
cars.column("Liability Insurence/Day",anchor=W,width=160)
cars.column("Comprehensive Insurence/Day",anchor=W,width=180)

cars.heading("#0",text="", anchor=W)
cars.heading("No.", text="No.", anchor=W)
cars.heading("Model",text="Model",anchor=W)
cars.heading("Availibility",text="Availibility",anchor=W)
cars.heading("Price/Day",text="Price/Day",anchor=W)
cars.heading("Liability Insurence/Day",text="Liability Insurence/Day",anchor=W)
cars.heading("Comprehensive Insurence/Day",text="Comprehensive Insurence/Day",anchor=W)

cars.insert(parent='',index='end',iid=0,text='',
values=('1','Silvia','1','250', '20', '200'))
cars.insert(parent='',index='end',iid=1,text='',
values=('2','Supra','4','300', '100', '250'))
cars.insert(parent='',index='end',iid=2,text='',
values=('3','R34','2','310', '100', '250'))
cars.insert(parent='',index='end',iid=3,text='',
values=('4','JZX100','5','200', '50', '200'))
cars.insert(parent='',index='end',iid=4,text='',
values=('5','RX-7','3','270', '100', '250'))
cars.insert(parent='',index='end',iid=5,text='',
values=('6','Hellcat','5','200', '70', '150'))



sl = Label( text = "Select one Option")
sl.config(font =("Times", 16))
def selectopt(sl):
    sl.configure(bg="light blue")
    sl.config(foreground="green")
    sl.pack()
# using place method we can set the position of label
    sl.place(relx = 0.05, rely = 0.4, anchor ='w')

selectopt(sl)


def show():
    newWindow1 = Toplevel(root)
    
    Label(newWindow1, text = "Car Return",font =("Times", 20),background="light blue", foreground="green").place(relx = 0.5, rely = 0.1, anchor ='center')
 
    # sets the title of the
    # Toplevel widget
    newWindow1.title("Car Return")
 
    # sets the geometry of toplevel
    newWindow1.geometry("350x350")
    newWindow1.config(bg = 'light blue')
  
    # Dropdown menu options
    options = ["Silvia","Supra","R34","JZX-100","RX-7","Hellcat"]
  
    # datatype of menu text
    clicked = StringVar()
  
    # initial menu text
    clicked.set( "Select Car" )
  
    # Create Dropdown menu
    drop = OptionMenu( newWindow1 , clicked, *options ).place(relx = 0.5, rely = 0.4, anchor ='center')
  
    # Create button, it will change label text
  
    # Create Label
    label = Label( newWindow1 , text ="", background="light blue", foreground= "Red", font=("Times",12))
    label.place(relx = 0.5, rely = 0.7, anchor ='center')

    def butn():
        label.config( text = clicked.get() )
        Label( newWindow1 , text ="Thank you for using BDL Enterprise you have returned: ", background="light blue", foreground= "Red", font=("Times",12)).place(relx = 0.5, rely = 0.6, anchor ='center')
    button = Button( newWindow1 , text = "Return" ,background='light green', command = butn ).place(relx = 0.5, rely = 0.5, anchor ='center')


    

crt = Button(root, text="Car Return", command= show )
crt.config(font = ("Times", 16))

def carreturn(crt):
    crt.configure(bg = "dark blue")
    crt.config(foreground="green")
    crt.pack()
# using place method we can set the position of label
    crt.place(relx = 0.05, rely = 0.65, anchor ='w')

carreturn(crt)


def tfd():
    newWindow2 = Toplevel(root)
    
    Label(newWindow2, text = "Total Financial Detail",font =("Times", 20),background="light blue", foreground="green").place(relx = 0.5, rely = 0.1, anchor ='center')
 
    # sets the title of the
    # Toplevel widget
    newWindow2.title("Total Financial Detail")
 
    # sets the geometry of toplevel
    newWindow2.geometry("600x550")
    newWindow2.config(bg = 'light blue')
    

    cartd1 = Frame()

    cars1 = ttk.Treeview(newWindow2) 


    cars1['columns'] = ('No.', 'Model', 'Availibility', 'Price/Day', 'Liability Insurence/Day', 'Comprehensive Insurence/Day')

    cars1.column("#0", width=0,  stretch=NO)
    cars1.column('No.', anchor=W, width=30)
    cars1.column("Model",anchor=W, width=55)
    cars1.column("Availibility",anchor=W,width=70)
    cars1.column("Price/Day",anchor=W,width=70)
    cars1.column("Liability Insurence/Day",anchor=W,width=140)
    cars1.column("Comprehensive Insurence/Day",anchor=W,width=170)

    cars1.heading("#0",text="", anchor=W)
    cars1.heading("No.", text="No.", anchor=W)
    cars1.heading("Model",text="Model",anchor=W)
    cars1.heading("Availibility",text="Availibility",anchor=W)
    cars1.heading("Price/Day",text="Price/Day",anchor=W)
    cars1.heading("Liability Insurence/Day",text="Liability Insurence/Day",anchor=W)
    cars1.heading("Comprehensive Insurence/Day",text="Comprehensive Insurence/Day",anchor=W)

    cars1.insert(parent='',index='end',iid=0,text='',
    values=('1','Silvia','1','300', '100', '250'))
    cars1.insert(parent='',index='end',iid=1,text='',
    values=('2','Supra','4','300', '100', '250'))
    cars1.insert(parent='',index='end',iid=2,text='',
    values=('3','R34','2','300', '100', '250'))
    cars1.insert(parent='',index='end',iid=3,text='',
    values=('4','JZX100','5','300', '100', '250'))
    cars1.insert(parent='',index='end',iid=4,text='',
    values=('5','RX-7','3','300', '100', '250'))
    cars1.insert(parent='',index='end',iid=5,text='',
    values=('6','Hellcat','5','300', '100', '250'))

    cartd1.place(relx = 0.65, rely = 0.65, anchor ='center')

    cars1.place(x = 0.6, y=0.6, anchor ='W')
    cars1.pack()

    Label(newWindow2, text = "Total Income : $997.5",font =("Times", 20),background="light blue", foreground="green").place(relx = 0.35, rely = 0.85, anchor ='center')
    Label(newWindow2, text = "Total Insurance : $350",font =("Times", 20),background="light blue", foreground="green").place(relx = 0.35, rely = 0.9, anchor ='center')    
    Label(newWindow2, text = "Total Tax : $47.5",font =("Times", 20),background="light blue", foreground="green").place(relx = 0.35, rely = 0.95, anchor ='center')


tfd = Button(root, text="Total Financial Detail", command=tfd)
tfd.config(font = ("Times", 16))
def finance(tfd):
    tfd.configure(bg = "dark blue")
    tfd.config(foreground="green")
    tfd.pack()
# using place method we can set the position of label
    tfd.place(relx = 0.05, rely = 0.8, anchor ='w')

finance(tfd)



#cr = Button(root, text="Car Rental", command=opentb)
#cr.config(font = ("Times", 16))
#def carrent(cr):
#    cr.configure(bg = "dark blue")
#    cr.config(foreground="green")
#    cr.pack()
# using place method we can set the position of label
#    cr.place(relx = 0.05, rely = 0.5, anchor ='w')

#carrent(cr)



def openNewWindow():
     
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(root)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("New Window")
 
    # sets the geometry of toplevel
    newWindow.geometry("1050x800")
    newWindow.config(bg = 'light blue')
    
    cartd = Frame()

    cars = ttk.Treeview(newWindow) 


    cars['columns'] = ('No.', 'Model', 'Availibility', 'Price/Day', 'Liability Insurence/Day', 'Comprehensive Insurence/Day')

    cars.column("#0", width=0,  stretch=NO)
    cars.column('No.', anchor=W, width=30)
    cars.column("Model",anchor=W, width=55)
    cars.column("Availibility",anchor=W,width=70)
    cars.column("Price/Day",anchor=W,width=70)
    cars.column("Liability Insurence/Day",anchor=W,width=140)
    cars.column("Comprehensive Insurence/Day",anchor=W,width=170)

    cars.heading("#0",text="", anchor=W)
    cars.heading("No.", text="No.", anchor=W)
    cars.heading("Model",text="Model",anchor=W)
    cars.heading("Availibility",text="Availibility",anchor=W)
    cars.heading("Price/Day",text="Price/Day",anchor=W)
    cars.heading("Liability Insurence/Day",text="Liability Insurence/Day",anchor=W)
    cars.heading("Comprehensive Insurence/Day",text="Comprehensive Insurence/Day",anchor=W)

    cars.insert(parent='',index='end',iid=0,text='',
    values=('1','Silvia','1','300', '100', '250'))
    cars.insert(parent='',index='end',iid=1,text='',
    values=('2','Supra','4','300', '100', '250'))
    cars.insert(parent='',index='end',iid=2,text='',
    values=('3','R34','2','300', '100', '250'))
    cars.insert(parent='',index='end',iid=3,text='',
    values=('4','JZX100','5','300', '100', '250'))
    cars.insert(parent='',index='end',iid=4,text='',
    values=('5','RX-7','3','300', '100', '250'))
    cars.insert(parent='',index='end',iid=5,text='',
    values=('6','Hellcat','5','300', '100', '250'))

    cartd.place(relx = 0.55, rely = 0.5, anchor ='center')

    cars.pack()
    cars.place(x= 510, y = 20)

    

    # A Label widget to show in toplevel
    Label(newWindow,text ="Summary", background='light blue', foreground='green',font=('Times',16)).place(x = 700, y = 300)
    Label(newWindow,text ="Car", background='light blue', foreground='green',font=('Times',16)).place(x= 510, y = 340 )
    Label(newWindow,text ="Rent Cost", background='light blue', foreground='green',font=('Times',16)).place(x= 510, y = 380)
    Label(newWindow,text ="Insurance Cost", background='light blue', foreground='green',font=('Times',16)).place(x= 510, y = 420)
    Label(newWindow,text ="Tax", background='light blue', foreground='green',font=('Times',16)).place(x = 510, y= 460)
    Label(newWindow,text ="--------------------------------------------------------------------", background='light blue', foreground='green',font=('Times',16)).place(x = 510, y= 500)
    Label(newWindow,text ="Total", background='light blue', foreground='green',font=('Times',16)).place(x = 510, y= 540)
    Label(newWindow,text ="--------------------------------------------------------------------", background='light blue', foreground='green',font=('Times',16)).place(x = 510, y= 270)

 #   inp = entry1.get()
 #   lbl.config(text = inp)
 #   lbl = tk.Label(newWindow, text = "")
 #   lbl.pack()
    v = StringVar(newWindow, "1")

    id1 = tk.Label(newWindow, text = "Select your car", background='light blue',foreground='red',font =("Times", 12)).place(x = 20, y = 10)
    def sel():
        str(var.get()) 
    
    var = IntVar()
    s1 = str()
    R1 = Radiobutton(newWindow, text="Silvia",background="light blue", foreground="red",font=("Times",12), variable=var, value=6, command=sel)
    R1.place(x= 40, y= 40)
    
    R2 = Radiobutton(newWindow, text="Supra",background="light blue", foreground="red",font=("Times",12), variable=var, value=7, command=sel)
    R2.place(x= 40, y= 70)

    R3 = Radiobutton(newWindow, text="R34",background="light blue", foreground="red",font=("Times",12), variable=var, value=8, command=sel)
    R3.place(x= 40, y= 100)

    R4 = Radiobutton(newWindow, text="JZX-100",background="light blue", foreground="red",font=("Times",12), variable=var, value=9, command=sel)
    R4.place(x= 40, y= 130)

    R5 = Radiobutton(newWindow, text="RX-7",background="light blue", foreground="red",font=("Times",12), variable=var, value=10, command=sel)
    R5.place(x= 40, y= 160)

    R6 = Radiobutton(newWindow, text="Hellcat",background="light blue", foreground="red",font=("Times",12), variable=var, value=11, command=sel)
    R6.place(x= 40, y= 190)

    label = Label(root)
    label.pack()
 
    nod = tk.Label(newWindow, text = "Number of Days",background='light blue',foreground='red',font =("Times", 12)).place(x = 20, y = 230)
    var1 = IntVar()
    d1 = Radiobutton(newWindow, text="1 Day",background="light blue", foreground="red",font=("Times",12), variable=var1, value=1, command=sel)
    d1.place(x= 40, y= 260)
    
    d2 = Radiobutton(newWindow, text="2 Days",background="light blue", foreground="red",font=("Times",12), variable=var1, value=2, command=sel)
    d2.place(x= 40, y= 290)

    d3 = Radiobutton(newWindow, text="3 Days",background="light blue", foreground="red",font=("Times",12), variable=var1, value=3, command=sel)
    d3.place(x= 40, y= 320)

    d4 = Radiobutton(newWindow, text="4 Days",background="light blue", foreground="red",font=("Times",12), variable=var1, value=4, command=sel)
    d4.place(x= 40, y= 350)

    d5 = Radiobutton(newWindow, text="5 Days",background="light blue", foreground="red",font=("Times",12), variable=var1, value=5, command=sel)
    d5.place(x= 40, y= 380)


    insurance = tk.Label(newWindow, text = "Which insurance do you want", background='light blue',foreground='red',font =("Times", 12)).place(x = 20, y = 430)
    var2 = IntVar()
    i1 = Radiobutton(newWindow, text="Liability Insurance",background="light blue", foreground="red",font=("Times",12), variable=var2, value=12, command=sel)
    i1.place(x= 40, y= 460)
    
    i2 = Radiobutton(newWindow, text="Comprehensive Insurance",background="light blue", foreground="red",font=("Times",12), variable=var2, value=13, command=sel)
    i2.place(x= 40, y= 490)
        

        #////////////////////////////////////////////////////////////////////////////
        # if statement for printing car name 
        
    def sum():
        if var == 6:
            Label(newWindow, text = "Silvia", font=('Times',16),background="light blue", foreground="green").place(x =950 , y = 400)
            if var1 == 1 and var2 == 1:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 420, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 1 and var2 == 2:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 557.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)

            elif var1 == 2 and var2 == 1:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 735, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 2 and var2 == 2:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 892.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
                
            elif var1 == 3 and var2 == 1:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1050, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 3 and var2 == 2:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1207.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            
            elif var1 == 4 and var2 == 1:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1365, font=('Times',16), foreground="green").place(x = 950 , y = 450)
            elif var1 == 4 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1522.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 5 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1837.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)      

        elif var == 7:
            Label(newWindow, text = "Supra", font=('Times',16),background="light blue", foreground="green").place(x =250 , y = 300)
            if var1 == 1 and var2 == 1:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 420, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 1 and var2 == 2:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 557.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)

            elif var1 == 2 and var2 == 1:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 735, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 2 and var2 == 2:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 892.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
                
            elif var1 == 3 and var2 == 1:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1050, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 3 and var2 == 2:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1207.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            
            elif var1 == 4 and var2 == 1:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1365, font=('Times',16), foreground="green").place(x = 950 , y = 450)
            elif var1 == 4 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1522.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 5 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1837.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)      

        elif var == 8:
            Label(newWindow, text = "R34", font=('Times',16),background="light blue", foreground="green").place(x =250 , y = 300)
            if var1 == 1 and var2 == 1:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 420, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 1 and var2 == 2:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 557.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)

            elif var1 == 2 and var2 == 1:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 735, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 2 and var2 == 2:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 892.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
                
            elif var1 == 3 and var2 == 1:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1050, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 3 and var2 == 2:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1207.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            
            elif var1 == 4 and var2 == 1:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1365, font=('Times',16), foreground="green").place(x = 950 , y = 450)
            elif var1 == 4 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1522.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 5 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1837.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)      
        elif var == 9:
            Label(newWindow, text = "JZX-100", font=('Times',16),background="light blue", foreground="green").place(x =250 , y = 300)
            if var1 == 1 and var2 == 1:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 420, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 1 and var2 == 2:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 557.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)

            elif var1 == 2 and var2 == 1:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 735, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 2 and var2 == 2:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 892.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
                
            elif var1 == 3 and var2 == 1:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1050, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 3 and var2 == 2:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1207.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            
            elif var1 == 4 and var2 == 1:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1365, font=('Times',16), foreground="green").place(x = 950 , y = 450)
            elif var1 == 4 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1522.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 5 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1837.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)      
        elif var == 10:
            Label(newWindow, text = "RX-7", font=('Times',16),background="light blue", foreground="green").place(x =250 , y = 300)
            if var1 == 1 and var2 == 1:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 420, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 1 and var2 == 2:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 557.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)

            elif var1 == 2 and var2 == 1:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 735, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 2 and var2 == 2:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 892.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
                
            elif var1 == 3 and var2 == 1:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1050, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 3 and var2 == 2:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1207.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            
            elif var1 == 4 and var2 == 1:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1365, font=('Times',16), foreground="green").place(x = 950 , y = 450)
            elif var1 == 4 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1522.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 5 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1837.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)      
        elif var == 11:
            Label(newWindow, text = "Hellcat", font=('Times',16),background="light blue", foreground="green").place(x =250 , y = 300)
            if var1 == 1 and var2 == 1:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 420, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 1 and var2 == 2:
                Label(newWindow, text = 300, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 557.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)

            elif var1 == 2 and var2 == 1:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 735, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 2 and var2 == 2:
                Label(newWindow, text = 600, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 892.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
                
            elif var1 == 3 and var2 == 1:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1050, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 3 and var2 == 2:
                Label(newWindow, text = 900, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1207.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            
            elif var1 == 4 and var2 == 1:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1365, font=('Times',16), foreground="green").place(x = 950 , y = 450)
            elif var1 == 4 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1522.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)
            elif var1 == 5 and var2 == 2:
                Label(newWindow, text = 1200, font=('Times',16), foreground="green").place(x = 950 , y = 450)
                Label(newWindow, text = 1837.5, font=('Times',16), foreground="green").place(x = 950 , y = 550)      


    Calculate = Button(newWindow, text="Calculate", command=sum)
    Calculate.config(font = ("Times", 12))
    Calculate.configure(bg = "dark blue")
    Calculate.config(foreground="green")
    Calculate.place(x = 430, y = 540)


ok = Button(root, text="Car Rental", command = openNewWindow)
ok.config(font = ("Times", 16))
ok.place(relx = 0.05, rely = 0.5, anchor ='w')
def okbtn(ok):
    ok.configure(bg = "dark blue")
    ok.config(foreground="green")
    cars.pack()
    
# using place method we can set the position of label
    
okbtn(ok)




root.mainloop()