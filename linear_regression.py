import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


#Load dataset
diabetes=load_diabetes()


#Select one feature(BMI=3rd column in dataset)
x=diabetes.data[:,np.newaxis,2]
y=diabetes.target


X_train,X_test,Y_train,Y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

model=LinearRegression()
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

print("Intercept (b0) : ",model.intercept_)
print("Coefficient (b1) : ",model.coef_[0])
print(f"Regression Line Equation : y ={model.intercept_:.2f} + {model.coef_[0]:.2f}*x")

#Evaluation

print("Mean Squared Error : ",mean_squared_error(Y_test,Y_pred))
print("R^2 Score : ",r2_score(Y_test,Y_pred))

#Vizualization

plt.scatter(X_test,Y_test,color='blue',label='Actual Data')
plt.plot(X_test,Y_pred,color='red',linewidth=2,label='regression line')
plt.xlabel("BMI:Body Mass Index")
plt.ylabel("Disease progression")
plt.title("Simple Linear Regression ")
plt.legend()
plt.show()