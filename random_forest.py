import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # take only first 2 features (sepal length, sepal width) for 2D plotting
y = iris.target

# 2. Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

# 5. Plot training points
for i, color in zip(range(3), ["red", "green", "blue"]):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], label=iris.target_names[i], color=color, edgecolor="k")

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("🌲 Random Forest Classification on Iris Dataset")
plt.legend()
plt.show()
