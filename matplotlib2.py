#1)
import matplotlib.pyplot as plt

days=['mon','Tue','Wed','Thu','Fri','Sat','Sun']
temps=[30,32,31,29,28,27,26]
plt.plot(days,temps,marker='o')
plt.title("Weekely temperature")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()

#2

months=['Jan','Feb','Mar','Apr','May']
productA=[200,240,210,300,280]
plt.bar(months,productA,color='blue')

plt.title("Product A Sales Over 5 Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

#3

days=['mon','Tue','Wed','Thu','Fri','Sat','Sun']
apple = [120, 125, 123, 130, 128, 132, 129]
google = [1000, 1005, 1010, 1020, 1015, 1030, 1025]
amazon = [1800, 1810, 1790, 1820, 1815, 1830, 1825]
plt.plot(days, apple, label='Apple')
plt.plot(days, google, label='Google')
plt.plot(days, amazon, label='Amazon')
plt.legend()
plt.title("Stock Prices Over a Week")
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()

#4

cities = ['City A', 'City B', 'City C', 'City D']
population = [150000, 120000, 180000, 100000]
plt.barh(cities, population, color='orange')
plt.title("Population by City")
plt.xlabel("Population")
plt.ylabel("City")
plt.show()

#5

brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Others']
share = [30, 25, 20, 10, 15]
plt.pie(share, labels=brands, autopct='%1.1f%%', startangle=140)
plt.title("Smartphone Market Share")
plt.axis('equal')
plt.show()

#6

scores = [56, 67, 45, 88, 72, 90, 61, 76, 84, 69]
plt.hist(scores, bins=5, color='green', edgecolor='black')
plt.title("Score Distribution")
plt.xlabel("Scores")
plt.ylabel("Number of Students")
plt.show()

#7

activities = ['Sleep', 'Work', 'Exercise', 'Leisure', 'Others']
time = [8, 9, 1, 4, 2]
plt.pie(time, labels=activities, autopct='%1.1f%%')
plt.title("Daily Time Distribution")
plt.axis('equal')
plt.show()

#8

height = [150, 155, 160, 165, 170, 175]
weight = [45, 50, 55, 60, 65, 70]
plt.scatter(height, weight, color='red')
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.grid(True)
plt.show()

#9

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
expenses = [2000, 2200, 2100, 2500, 2400]
plt.fill_between(months, expenses, color='orange', alpha=1)
plt.plot(months, expenses, color='blue')
plt.title("Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Expense (INR)")
plt.show()

#10

x = [10, 20, 30, 40, 50]
y = [15, 25, 35, 45, 55]
sizes = [100, 300, 500, 700, 900]
plt.scatter(x, y, s=sizes, alpha=0.5, c='purple')
plt.title("Bubble Chart Example")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

#11

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.title("Saved Plot")
plt.savefig("plot_output.png", dpi=300)
plt.show()