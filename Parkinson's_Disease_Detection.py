# IMPORTING DEPENDENCIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Display all columns and rows for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ===========================
# DATA COLLECTION AND ANALYSIS
# ===========================

# Load the dataset
data_path = "D:\\pythonProject4\\parkinsons.csv"
data = pd.read_csv(data_path)

# Display the first five rows of the dataset
print("\nSample Data:")
print(data.head())

# Display basic information about the dataset
print("\nDataset Information:")
print(data.info())

# Display statistical summary of the dataset
print("\nStatistical Summary:")
print(data.describe())

# Check the distribution of the target variable ('status')
# 1 => People with Parkinson's Disease, 0 => Healthy People
print("\nTarget Distribution:")
print(data['status'].value_counts())

# ===========================
# DATA PREPROCESSING
# ===========================

# Grouping the data based on the target variable for analysis
print("\nAverage Feature Values by Status:")
print(data.groupby('status').mean())

# Separate features and target variable
X = data.drop(columns=['name', 'status'], axis=1)
Y = data['status']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the feature data for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================
# MODEL TRAINING
# ===========================

# Create and train the Support Vector Machine (SVM) model with a linear kernel
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# ===========================
# MODEL EVALUATION
# ===========================

# Predict on training data and calculate accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"\nTraining Data Accuracy: {train_accuracy:.2f}")

# Predict on test data and calculate accuracy
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test Data Accuracy: {test_accuracy:.2f}")

# ===========================
# BUILDING A PREDICTIVE SYSTEM
# ===========================

# Take user input for prediction
input_data = input("\nEnter the report details separated by commas:\n")

# Convert user input into a tuple of floats
input_data = tuple(map(float, input_data.split(',')))

# Convert input data to numpy array and reshape for prediction
input_data_as_array = np.asarray(input_data).reshape(1, -1)

# Standardize input data
std_input_data = scaler.transform(input_data_as_array)

# Make prediction
prediction = model.predict(std_input_data)[0]
result = "Parkinson's Disease Detected" if prediction == 1 else "Healthy"

print(f"\nPrediction Result: {result}")

# ===========================
# DECISION FUNCTION ANALYSIS (Optional)
# ===========================

# Get decision scores to analyze model confidence
decision_scores = model.decision_function(X_test)

# Convert scores to binary predictions based on cutoff
cutoff = 0
predicted_labels = (decision_scores > cutoff).astype(int)

# Evaluate prediction accuracy from decision scores
score_based_accuracy = accuracy_score(Y_test, predicted_labels)
print(f"\nDecision Function Accuracy: {score_based_accuracy:.2f}")

# ===========================
# FINAL OUTPUT
# ===========================

if prediction == 0:
    print("\n✅ The person is HEALTHY and not showing signs of Parkinson's Disease.")
else:
    print("\n🚨 The person is showing signs of PARKINSON'S DISEASE.")
