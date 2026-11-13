import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# open file with pd.read_csv ('data.csv' or 'Iris.csv')
df = pd.read_csv('data.csv')
print("Shape of dataset (rows, columns)", df.shape)

# print head of data set
print("\n **************First 5 rows of dataset**********\n", df.head())

# Select features and target based on dataset
if 'Species' in df.columns:
    drop_cols = [col for col in ['Id', 'Species'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['Species']
    target_names = sorted(df['Species'].unique())
elif 'Outcome' in df.columns:
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    target_names = ['Normal', 'Diabetes']
elif 'diagnosis' in df.columns:
    drop_cols = [col for col in ['id', 'diagnosis', 'Unnamed: 32'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['diagnosis']
    target_names = ['Benign (B)', 'Malignant (M)']
else:
    raise ValueError('Target column not recognized in dataset')

print("\n Features set\n", X.head())
print("\n Target labels\n", y.head())

# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# random forest model creation
rfc = RandomForestClassifier(n_estimators=3, max_features=5, random_state=42)
rfc.fit(X_train, y_train)

# predictions
rfc_predict = rfc.predict(X_test)

# Evaluation
# 1. Accuracy Score
accuracy = accuracy_score(y_test, rfc_predict)
print("=== Accuracy Score ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

# 2. Confusion Matrix
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')

# 3. Classification Report
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict, target_names=target_names))

# Visualizing True vs Predicted values
index = np.arange(0, len(y_test))
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.scatter(index, y_test, c="red", label='True Value', alpha=0.7)
plt.scatter(index, rfc_predict, c="blue", label='Predicted Value', marker='x')
plt.title(f'Random Forest Classifier: True vs Predicted (Accuracy: {accuracy * 100:.2f}%)')
plt.xlabel('Sample Index')
plt.ylabel('Diagnosis')
plt.legend()
plt.show()