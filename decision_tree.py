import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load the Iris dataset
iris = load_iris()

# 2. Train a Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)  # limit depth for clarity
dt_model.fit(iris.data, iris.target)

# 3. Plot the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
