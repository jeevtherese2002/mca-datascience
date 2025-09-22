import os
import pandas as pd
path=os.path.expanduser('~/Downloads/employee_data.csv')
file=pd.read_csv(path)

#1
print(file)
#2
print(file.dtypes)
#3
print(file.describe())
#4
high_salary = file[file['Salary'] > 50000]
print(high_salary)
#5
hr_emp=file[file['Department']=='HR']['Name']
print(hr_emp)
#6
file['Tax'] = file['Salary'] * 0.10
print(file[['Name', 'Salary', 'Tax']])
#7
avg_salary=file.groupby('Department')['Salary'].mean()
print("average salary of the department: ")
print(avg_salary)
#8
sorted_age=file.sort_values(by='Age',ascending=False)
print(sorted_age)
#9
cleaned_file=file.dropna()
print("DataFrAME AFTER DROPPING ROWS WITH MISSING VALUES:")
print(cleaned_file)
#10
cleaned_file.to_csv('cleaned_data.csv',index=False)
print("cleaned dataset saved as 'cleaned_employees.csv'")
#11
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.bar(file
['Name'],file
['Salary'],color='skyblue')
plt.title('Employee Salaries')
plt.xlabel('Employee Name')
plt.ylabel('Salary (INR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#12
department_counts=file['Department'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(department_counts,labels=department_counts.index,autopct='%1.1f%%',startangle=90)
plt.title('Distribution of Employees by Department')
plt.show()