# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('exp1.csv')

# Step 2: Preprocess the data
X = data.drop('power_mode', axis=1)
y = data['power_mode']

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = np.round(lin_reg.predict(X_test))  # Rounding predictions for classification
y_pred = np.clip(y_pred, 0, len(np.unique(y)) - 1)  # Ensure predictions are within valid class range

print("\nClassification Report:")
print(classification_report(y_test, y_pred.astype(int)))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred.astype(int))
print(conf_matrix)

# Visualize Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Linear Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
