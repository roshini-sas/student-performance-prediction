import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")
print("Dataset loaded successfully ")

# Preview data
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nShape (rows, columns):")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Data types
print("\nData types:")
print(df.dtypes)
# STEP 1: Convert categorical text to numbers
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['previous_grade'] = le.fit_transform(df['previous_grade'])
df['result'] = le.fit_transform(df['result'])

print("\nAfter encoding categorical columns:")
print(df)
# STEP 2: Split features and target
X = df[['attendance', 'study_hours', 'internal_marks', 'previous_grade']]
y = df['result']

print("\nInput features (X):")
print(X)

print("\nTarget output (y):")
print(y)
# STEP 3: Train a Machine Learning model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel accuracy:", accuracy)
# STEP 4: Predict result for a new student
# Example: attendance, study_hours, internal_marks, previous_grade

new_student = [[80, 3, 70, 1]]  

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nPrediction: Student will PASS ")
else:
    print("\nPrediction: Student will FAIL ")
