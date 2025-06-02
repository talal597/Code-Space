
----------------------------------------------INTERESTING QUERIES--------------------------------------------------------------
--1*
--List of Active Trains with Their Current Schedules:
SELECT Trains.Train_ID, Trains.Train_Name, Schedules.Departure_Time, Schedules.Arrival_Time
FROM Trains
INNER JOIN Schedules ON Trains.Train_ID = Schedules.Train_ID
WHERE Trains.Train_Status = 'Active';

--2*
--List of Trains with Their Crew Members and Assigned Roles:
SELECT Trains.Train_Name, TrainCrew.Crew_FName, TrainCrew.Crew_LName, TrainCrew.Crew_Role
FROM Trains
INNER JOIN TrainCrew ON Trains.Train_ID = TrainCrew.Train_ID;

--3*
--Details of Maintenance for Each Train Type:
SELECT Trains.Train_Type, Maintenance.Maintenance_ID, Maintenance.Maintenance_Date, Maintenance.Maintenance_Type
FROM Maintenance
INNER JOIN Trains ON Maintenance.Train_ID = Trains.Train_ID;

--4*
--Average Salary of Employees in Each Department:
SELECT Departments.Department_Name, AVG(Employees.Employee_Salary) AS Avg_Salary
FROM Employees
INNER JOIN Departments ON Employees.Employee_Department = Departments.Department_ID
GROUP BY Departments.Department_Name;

--6*
--Top 5 Trains with the most tickets sold
SELECT TOP 5 Tickets.Ticket_Train, Trains.Train_Name, COUNT(*) AS Total_Tickets_Sold
FROM Tickets
INNER JOIN Trains ON Tickets.Ticket_Train = Trains.Train_ID
GROUP BY Tickets.Ticket_Train, Trains.Train_Name
ORDER BY Total_Tickets_Sold DESC;

--7*
--Revenue Per Payment Methood
SELECT Payment_Method, SUM(Transaction_Amount) AS Total_Revenue
FROM RevenueTransaction
GROUP BY Payment_Method;

--8*
--Passangers Whose Names Start with 'A'
SELECT Passanger_FName, Passanger_LName
FROM Passangers
WHERE Passanger_FName LIKE 'A%';

--9
--Tickets Costing more than 4800
SELECT Ticket_ID, Ticket_Train, Ticket_price
FROM Tickets
WHERE Ticket_price > 4800;

--10
--Details of Passenger Tickets Booked for a Specific Train:
SELECT Tickets.Ticket_ID, Passengers.Passenger_FName, Passengers.Passenger_LName, Tickets.Ticket_Date
FROM Tickets
INNER JOIN Passengers ON Tickets.Ticket_Passenger = Passengers.Passenger_ID
WHERE Tickets.Ticket_Train = 'T004' AND Tickets.Ticket_Status = 'Confirmed';

--11
--List of Passengers with Their Booked Tickets and Departure Station:
SELECT Passengers.Passenger_FName, Passengers.Passenger_LName, Tickets.Ticket_ID, Schedules.Departure_Time
FROM Passengers
INNER JOIN Tickets ON Passengers.Passenger_ID = Tickets.Ticket_Passenger
INNER JOIN Schedules ON Tickets.Ticket_Train = Schedules.Train_ID;

--12
--Total Revenue Generated from Ticket Sales for Each Train:
SELECT Tickets.Ticket_Train, SUM(Tickets.Ticket_Price) AS Total_Revenue
FROM Tickets
GROUP BY Tickets.Ticket_Train;

--13
--List of Routes with Total Distance Covered and Departure Time:
SELECT Routes.Route_From, Routes.Route_To, Routes.Route_Distance, Schedules.Departure_Time
FROM Routes
INNER JOIN Schedules ON Routes.Route_ID = Schedules.Route_ID;

--14
--Details of Passengers with Their Ticket Status and Departure Station
SELECT Passengers.Passenger_FName, Passengers.Passenger_LName, Tickets.Ticket_Status, Routes.Route_From , Route_To
FROM Passengers
INNER JOIN Tickets ON Passengers.Passenger_ID = Tickets.Ticket_Passenger
INNER JOIN Schedules ON Tickets.Ticket_Train = Schedules.Train_ID
INNER JOIN Routes ON Schedules.Route_ID = Routes.Route_ID;

--15
--List of Trains with Their Available Classes and Capacities
SELECT Trains.Train_Name, ServiceClass.Class_Name, ServiceClassAllocation.Class_Capacity
FROM Trains
INNER JOIN ServiceClassAllocation ON Trains.Train_ID = ServiceClassAllocation.Train_ID
INNER JOIN ServiceClass ON ServiceClassAllocation.Class_ID = ServiceClass.Class_ID;

--16
--List of Train Companies that pay their Employees mmore than 70,000
SELECT DISTINCT Trains.Train_Name
FROM Trains
JOIN TrainCrew ON Trains.Train_ID = TrainCrew.Train_ID
WHERE TrainCrew.Crew_Salary > 70000;

--17
--Top 5 Passangers with the most Bookings
SELECT TOP 5 Passangers.Passanger_ID, Passangers.Passanger_FName, Passangers.Passanger_LName, COUNT(Booking.Booking_ID) AS Total_Bookings
FROM Passangers
JOIN Booking ON Passangers.Passanger_ID = Booking.Passanger_ID
GROUP BY Passangers.Passanger_ID, Passangers.Passanger_FName, Passangers.Passanger_LName
ORDER BY Total_Bookings DESC;


