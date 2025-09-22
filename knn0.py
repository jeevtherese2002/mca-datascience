from sklearn . datasets import load_iris
import pandas as pd
# Load the dataset
iris = load_iris ()
# Create a DataFrame for pretty tabular view
df = pd . DataFrame ( data = iris . data , columns = iris . feature_names )
df ['target '] = iris . target
df ['species '] = df ['target ']. apply ( lambda i : iris . target_names [ i ])
# Display first 10 rows
print ("\ nFirst 10 rows of the Iris dataset :\n")
print ( df . head (10) )