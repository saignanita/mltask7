  # Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # Import joblib to save the model

# Load the dataset
data = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\mltask6\student dropout.csv')

# Preprocess the data (encoding categorical variables)
X = data.drop(columns=['Dropped_Out'])  # Features
y = data['Dropped_Out'].astype(int)     # Target

# Label encoding for binary categorical features
binary_columns = ['Gender', 'Address', 'Family_Size', 'Parental_Status', 'School_Support', 
                  'Family_Support', 'Extra_Paid_Class', 'Extra_Curricular_Activities', 
                  'Attended_Nursery', 'Wants_Higher_Education', 'Internet_Access', 'In_Relationship']

le = LabelEncoder()
for col in binary_columns:
    X[col] = le.fit_transform(X[col])

# One-hot encoding for categorical features with multiple categories
X = pd.get_dummies(X, columns=['School', 'Mother_Job', 'Father_Job', 'Reason_for_Choosing_School', 'Guardian'], drop_first=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
print("Training Random Forest...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy for Random Forest: {accuracy_score(y_test, y_pred):.4f}")
print(f"Confusion Matrix for Random Forest:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report for Random Forest:\n{classification_report(y_test, y_pred)}")

# Save the trained model to a .pkl file
joblib.dump(model, r'C:\Users\LENOVO\OneDrive\Desktop\mltask6\random_forest_model.pkl')
print("Model saved as random_forest_model.pkl")
# After training the model
features = X.columns.tolist()  # Get the feature names
joblib.dump(features, r'C:\Users\LENOVO\OneDrive\Desktop\mltask6\model_features.pkl')
