import pandas as pd
from numpy.ma.extras import column_stack
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#Load datasets
df=pd.read_csv("play_tennis_dataset_500.csv")

#seperate features and target
x=df.drop("PlayTennis",axis=1)
y=df["PlayTennis"]

#Encode categorical columns
encoders={}
for col in x.columns:
    le=LabelEncoder()
    x[col]=le.fit_transform(x[col])
    encoders[col]=le


#Encode target variable
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)

#split datasets
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.26,random_state=42
)

#create naive bayes model for categorical data

model=CategoricalNB()
model.fit(x_train,y_train)

#predictions
y_pred=model.predict(x_test)

#Evaluation metrics

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="weighted")
recall=recall_score(y_test,y_pred,average="weighted")
f1=f1_score(y_test,y_pred,average="weighted")


#print neatly
print("Model performance matrics:")
print(f"Accuracy : {accuracy:4f}")
print(f"Precision : {precision:4f}")
print(f"Recall : {recall:4f}")
print(f"F1-score : {f1:4f}")