import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset (replace with the correct file path)
df = pd.read_csv(r'C:\Users\DELL\Desktop\titanic_pred\titanic.csv')


# Inspect dataset
print(df.head())
print(df.columns)

# Rename columns to match the feature selection logic
df.rename(
    columns={
        "Siblings/Spouses Aboard": "SibSp",
        "Parents/Children Aboard": "Parch"
    },
    inplace=True,
)

# Use the relevant columns
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Encode categorical variables
le_gender = LabelEncoder()
df['Sex'] = le_gender.fit_transform(df['Sex'])  # Male=1, Female=0

# Define features and target
X = df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'titanic_model.pkl')
print("Model saved as 'titanic_model.pkl'")
