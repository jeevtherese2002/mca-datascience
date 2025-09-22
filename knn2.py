from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
iris=load_iris()

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
df['species']=df['target'].apply(lambda i : iris.target_names[i])

print("    iris datasets-first 10 rows :\n ")
print(df.head(10))
print("\n  Features:",iris.faeture_names)
print("    Target Names:",list(iris.target_names))
print()

X=iris.data
y=iris.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)


knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print(f"   Model accuracy on Test Data: {accuracy * 100:.2f}%\n")

sample=[[6.0,2.9,4.5,1.5]]
predicted_class=knn.predict(sample)[0]
predicted_name=iris.target_name[predicted_class]

print(f"     Sample Input: {sample}")
print(f"     Protected Class Index :{predicted_class}")
print(f"     Predicted class name:{predicted_name}")