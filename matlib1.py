import matplotlib.pyplot as plt

days=['mon','Tue','Wed','Thu','Fri','Sat','Sun']
temps=[30,32,31,29,28,27,26]
plt.plot(days,temps,marker='o')
plt.title("Weekely temperature")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()

months=['Jan','Feb','Mar','Apr','May']
productA=[200,240,210,300,280]
plt.bar(months,productA,color='blue')

plt.title("Product A Sales Over 5 Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

import matplotlib.pyplot as plt
brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Others']
share = [30, 25, 20, 10, 15]
plt.pie(share, labels=brands, autopct='%1.1f%%', startangle=140)
plt.title("Smartphone Market Share")
plt.axis('equal')
plt.show()