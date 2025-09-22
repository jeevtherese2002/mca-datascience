# 1) Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 2) Load the CSV
# Replace with your actual path if different
csv_path = "play_tennis_dataset_500.csv"

df = pd.read_csv(csv_path)

# If your file does NOT have headers, uncomment the following and re-run:
# df = pd.read_csv(csv_path, header=None,
#                  names=["ID","Outlook","Temperature","Humidity","Wind","PlayTennis"])

# 3) Basic cleanup (drop an ID column if present)
for possible_id in ["ID","Day","Index","S.No","Sno","No"]:
    if possible_id in df.columns:
        df = df.drop(columns=[possible_id])
        break

# 4) Define features (X) and target (y)
#    We expect columns: Outlook, Temperature, Humidity, Wind, PlayTennis
target_col = "PlayTennis"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

# 5) Make features numeric
#    - Outlook: nominal (Sunny/Overcast/Rain) -> One-hot
#    - Temperature: ordinal Cool < Mild < Hot -> OrdinalEncoder
#    - Humidity: binary Normal/High -> OrdinalEncoder Normal=0, High=1
#    - Wind: binary Weak/Strong -> OrdinalEncoder Weak=0, Strong=1

# Identify columns by name (adapt if your headers differ)
outlook_col = "Outlook"
temp_col = "Temperature"
hum_col = "Humidity"
wind_col = "Wind"

# If your CSV used different names, set them here:
# outlook_col, temp_col, hum_col, wind_col = "Outlook","Temperature","Humidity","Wind"

preprocessor = ColumnTransformer(
    transformers=[
        # One-hot for Outlook (3 dummies)
        ("outlook_ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [outlook_col]),

        # Ordinal for Temperature with a meaningful order
        ("temp_ord", OrdinalEncoder(categories=[["Cool","Mild","Hot"]]), [temp_col]),

        # Ordinal/binary encodings (explicit category order controls mapping to 0/1)
        ("hum_ord", OrdinalEncoder(categories=[["Normal","High"]]), [hum_col]),
        ("wind_ord", OrdinalEncoder(categories=[["Weak","Strong"]]), [wind_col]),
    ],
    remainder="drop"
)

# 6) Build the pipeline: preprocessing -> GaussianNB
clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("gnb", GaussianNB())
])

# 7) Train/valid split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.26, random_state=42, stratify=y
)

# 8) Fit the model
clf.fit(X_train, y_train)

# 9) Predict & evaluate
y_pred = clf.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("Model performance metrics (GaussianNB on numeric features):")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}\n")

print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
