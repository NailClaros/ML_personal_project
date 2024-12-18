import subprocess
import sys

def install_requirements(requirements_file):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Successfully installed packages from {requirements_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
install_requirements("requirements.txt")

import kagglehub
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
##Loading data!
print("Loading data ----------------------------------")
path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
print("Path to dataset files:", path)
loan_data = pd.read_csv(f"{path}/loan_data.csv")
print(loan_data.info()) ## defining data and target values

'''
RangeIndex: 45000 entries, 0 to 44999
Data columns (total 14 columns):
 #   Column                          Non-Null Count  Dtype
---  ------                          --------------  -----
 0   person_age                      45000 non-null  float64     Age of the person
 1   person_gender                   45000 non-null  object      Gender of the person (Categorical)
 2   person_education                45000 non-null  object      Highest education level (Categorical)
 3   person_income                   45000 non-null  float64     Annual income
 4   person_emp_exp                  45000 non-null  int64       Years of employment experience
 5   person_home_ownership           45000 non-null  object      Home ownership status (e.g., rent, own, mortgage)
 6   loan_amnt                       45000 non-null  float64     Loan amount requested
 7   loan_intent                     45000 non-null  object      Purpose of the loan (Categorical)
 8   loan_int_rate                   45000 non-null  float64     Loan interest rate
 9   loan_percent_income             45000 non-null  float64     Loan amount as a percentage of annual income
 10  cb_person_cred_hist_length      45000 non-null  float64     Length of credit history in years
 11  credit_score                    45000 non-null  int64       Credit score of the person
 12  previous_loan_defaults_on_file  45000 non-null  object      Indicator of previous loan defaults (Categorical)
 13  loan_status                     45000 non-null  int64 <---- Loan approval status: 1 = approved; 0 = rejected
dtypes: float64(6), int64(3), object(5)
memory usage: 4.8+ MB
'''
print(loan_data.describe())
'''
         person_age  person_income  ...  credit_score   loan_status
count  45000.000000   4.500000e+04  ...  45000.000000  45000.000000
mean      27.764178   8.031905e+04  ...    632.608756      0.222222
std        6.045108   8.042250e+04  ...     50.435865      0.415744
min       20.000000   8.000000e+03  ...    390.000000      0.000000
25%       24.000000   4.720400e+04  ...    601.000000      0.000000
50%       26.000000   6.704800e+04  ...    640.000000      0.000000
75%       30.000000   9.578925e+04  ...    670.000000      0.000000
max      144.000000   7.200766e+06  ...    850.000000      1.000000
'''

print(loan_data.head())

##Cleaning data!
print("cleaning data ----------------------------------")
##first is handling null values
null_values = loan_data.isnull().sum()

# Print columns with null values (if any)
print("Null values in each column:")
print(null_values)

# Highlight columns with null values
columns_with_null = null_values[null_values > 0]
if not columns_with_null.empty:
    print("\nColumns with null values:")
    print(columns_with_null)
else:
    print("\nNo null values found in the dataset!")

##Now we check for wrong value types within columns
##We first make a dictionary on the column types, called expected types
expected_types = {
    "person_age": "float64",
    "person_gender": "object",
    "person_education": "object",
    "person_income": "float64",
    "person_emp_exp": "int64",
    "person_home_ownership": "object",
    "loan_amnt": "float64",
    "loan_intent": "object",
    "loan_int_rate": "float64",
    "loan_percent_income": "float64",
    "cb_person_cred_hist_length": "float64",
    "credit_score": "int64",
    "previous_loan_defaults_on_file": "object",
    "loan_status": "int64",
}
##we will loop through each column and check each row to ensure the value is of the 
##correct value type with respect to each columb

##all incorrect values go into the mismatched_types dictioanary
mismatched_types = {}
for column, expected_type in expected_types.items():
    actual_type = loan_data[column].dtype
    if actual_type != expected_type:
        mismatched_types[column] = {"expected": expected_type, "actual": actual_type}

#display if there are any types we need to fix
if mismatched_types:
    print("Columns with mismatched data types:")
    for column, types in mismatched_types.items():
        print(f" - {column}: Expected {types['expected']}, but found {types['actual']}")
else:
    print("All columns have the correct data types!\n\n")

##Now that we have checked for no inconsistent data/nulls, we will fix categorical data and have a look at what unique values will need to be fixed
#we will also look at the target to ensure all values are 0 or 1
categorical_columns = loan_data.select_dtypes(include=["object"]).columns
categorical_columns = [col for col in categorical_columns]
categorical_columns.append("loan_status")
# Print unique values for each categorical column
print("Unique values for categorical columns:")
for column in categorical_columns:
    unique_values = loan_data[column].unique()
    print(f"{column}: {unique_values}")

'''
person_gender: ['female' 'male']
person_education: ['Master' 'High School' 'Bachelor' 'Associate' 'Doctorate']
person_home_ownership: ['RENT' 'OWN' 'MORTGAGE' 'OTHER']
loan_intent: ['PERSONAL' 'EDUCATION' 'MEDICAL' 'VENTURE' 'HOMEIMPROVEMENT' 'DEBTCONSOLIDATION']
previous_loan_defaults_on_file: ['No' 'Yes']
loan_status: [1 0]

'''

###Now that we have found all categorical values and everything appears to be in good condition we will manipluate the data
##to function for 2 models. Logistic regression and Random Forest
RF_loan_data = loan_data.copy(deep=True)#this one is for Random Forest
LR_loan_data = loan_data.copy(deep=True)#this one is for Logisctic regression - requires data normalization

##We shall start with Logistic regression and then return to do Random Forest

categorical_columns = loan_data.select_dtypes(include=["object"]).columns
print("\nLogistic Regression-------------------------------------------------------------")
print("\nUnique values for categorical columns after encoding------------:")
for column in categorical_columns:
    # Convert column to categorical type
    LR_loan_data[column] = LR_loan_data[column].astype("category")
    # Assign category codes to the column
    LR_loan_data[column] = LR_loan_data[column].cat.codes
    # Print unique codes for the column
    unique_values = LR_loan_data[column].unique()
    print(f"{column}: {unique_values}")

'''
Unique values for categorical columns:
person_gender: [0 1] === ['female' 'male']
person_education: [4 3 1 0 2] === ['Master' 'High School' 'Bachelor' 'Associate' 'Doctorate']
                    0 = Associate
                    1 = Bachelor
                    2 = Doctorate
                    3 = High School
                    4 = Master
person_home_ownership: [3 2 0 1] === ['RENT' 'OWN' 'MORTGAGE' 'OTHER']
                    0 = MORTGAGE
                    1 = OTHER
                    2 = OWN
                    3 = RENT
loan_intent: [4 1 3 5 2 0] === ['PERSONAL' 'EDUCATION' 'MEDICAL' 'VENTURE' 'HOMEIMPROVEMENT' 'DEBTCONSOLIDATION']
                    0 = DEBTCONSOLIDATION
                    1 = EDUCATION
                    2 = HOMEIMPROVEMENT
                    3 = MEDICAL
                    4 = PERSONAL
                    5 = VENTURE
previous_loan_defaults_on_file: [0 1] === ['No' 'Yes']
                    0 = No
                    1 = Yes
'''
print(LR_loan_data.head())
##After converting data points to numeric, we can do display a correlation matrix
correlation_matrix = LR_loan_data.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Heat Map of Loan data (via LR_load_data)")
plt.show()

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
x = LR_loan_data.drop(columns=['loan_status'])  
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x) #scaling X for logistic regression
X_scaled= pd.DataFrame(X_scaled, columns=x.columns)

y = LR_loan_data['loan_status']  
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=19)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000000)

logistic_model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

print("\n\nhyper param tunning via GridSearchCV---------------------------------------------------------")
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'C': [0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear']},  # For liblinear
    {'C': [0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']},  # For lbfgs
]

grid_search = GridSearchCV(
    logistic_model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_  

# Evaluate on the test set
y_test_pred = best_model.predict(X_test)

print("Test Set Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Select a random sample from the dataset
rejected_loans = LR_loan_data[LR_loan_data['loan_status'] == 0]
approved_loans = LR_loan_data[LR_loan_data['loan_status'] == 1]

# Randomly sample 5 from each group
sample_rejected = rejected_loans.sample(5, random_state=19)
sample_approved = approved_loans.sample(5, random_state=19)

# Combine the samples
sample_data = pd.concat([sample_rejected, sample_approved])

# Extract the actual values (loan_status)
actual_vals = sample_data['loan_status']

# Drop the loan_status column for predictions
sample_data = sample_data.drop(columns=['loan_status'])

# Predict the outcomes using the trained model
predictions = best_model.predict(sample_data)

# Display the results
print("\nPredictions vs Actual Values---------------------")
for i in range(len(sample_data)):
    print(f"Sample {i+1}:")
    print(f"Input Features:\n{sample_data.iloc[i]}")
    print(f"Predicted Loan Status (0 = Rejected, 1 = Approved): {predictions[i]}")
    print(f"Actual Loan Status: {actual_vals.iloc[i]}")
    print("-" * 50)

'''
Predictions vs Actual Values---------------------
Sample 1:
Input Features:
person_age                            28.00
person_gender                          0.00
person_education                       3.00
person_income                     120915.00
person_emp_exp                         7.00
person_home_ownership                  3.00
loan_amnt                          18000.00
loan_intent                            3.00
loan_int_rate                         13.91
loan_percent_income                    0.15
cb_person_cred_hist_length             9.00
credit_score                         501.00
previous_loan_defaults_on_file         1.00
Name: 37968, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 2:
Input Features:
person_age                           22.00
person_gender                         1.00
person_education                      0.00
person_income                     73010.00
person_emp_exp                        0.00
person_home_ownership                 0.00
loan_amnt                         18000.00
loan_intent                           0.00
loan_int_rate                        12.99
loan_percent_income                   0.25
cb_person_cred_hist_length            2.00
credit_score                        521.00
previous_loan_defaults_on_file        1.00
Name: 8764, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 3:
Input Features:
person_age                           36.00
person_gender                         0.00
person_education                      3.00
person_income                     60601.00
person_emp_exp                       15.00
person_home_ownership                 3.00
loan_amnt                          3125.00
loan_intent                           2.00
loan_int_rate                        13.47
loan_percent_income                   0.05
cb_person_cred_hist_length           14.00
credit_score                        671.00
previous_loan_defaults_on_file        1.00
Name: 29672, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 1
Actual Loan Status: 0
--------------------------------------------------
Sample 4:
Input Features:
person_age                            27.00
person_gender                          1.00
person_education                       0.00
person_income                     115426.00
person_emp_exp                         0.00
person_home_ownership                  3.00
loan_amnt                          12000.00
loan_intent                            1.00
loan_int_rate                         11.11
loan_percent_income                    0.10
cb_person_cred_hist_length             6.00
credit_score                         612.00
previous_loan_defaults_on_file         0.00
Name: 25642, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 5:
Input Features:
person_age                           23.00
person_gender                         1.00
person_education                      0.00
person_income                     66193.00
person_emp_exp                        5.00
person_home_ownership                 3.00
loan_amnt                         15550.00
loan_intent                           5.00
loan_int_rate                        11.71
loan_percent_income                   0.23
cb_person_cred_hist_length            4.00
credit_score                        679.00
previous_loan_defaults_on_file        0.00
Name: 17504, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 6:
Input Features:
person_age                           24.00
person_gender                         1.00
person_education                      1.00
person_income                     72818.00
person_emp_exp                        0.00
person_home_ownership                 3.00
loan_amnt                          7000.00
loan_intent                           1.00
loan_int_rate                         7.51
loan_percent_income                   0.10
cb_person_cred_hist_length            4.00
credit_score                        662.00
previous_loan_defaults_on_file        0.00
Name: 7140, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 1
--------------------------------------------------
Sample 7:
Input Features:
person_age                           25.00
person_gender                         0.00
person_education                      1.00
person_income                     51103.00
person_emp_exp                        4.00
person_home_ownership                 3.00
loan_amnt                         18000.00
loan_intent                           5.00
loan_int_rate                         6.17
loan_percent_income                   0.35
cb_person_cred_hist_length            2.00
credit_score                        642.00
previous_loan_defaults_on_file        0.00
Name: 970, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 1
--------------------------------------------------
Sample 8:
Input Features:
person_age                            24.00
person_gender                          1.00
person_education                       3.00
person_income                     128248.00
person_emp_exp                         2.00
person_home_ownership                  3.00
loan_amnt                          13000.00
loan_intent                            2.00
loan_int_rate                         12.69
loan_percent_income                    0.10
cb_person_cred_hist_length             4.00
credit_score                         514.00
previous_loan_defaults_on_file         0.00
Name: 17357, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 1
--------------------------------------------------
Sample 9:
Input Features:
person_age                           26.00
person_gender                         0.00
person_education                      1.00
person_income                     65622.00
person_emp_exp                        3.00
person_home_ownership                 3.00
loan_amnt                         25000.00
loan_intent                           0.00
loan_int_rate                        16.52
loan_percent_income                   0.38
cb_person_cred_hist_length            4.00
credit_score                        555.00
previous_loan_defaults_on_file        0.00
Name: 43585, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 1
--------------------------------------------------
Sample 10:
Input Features:
person_age                           22.00
person_gender                         1.00
person_education                      3.00
person_income                     86016.00
person_emp_exp                        0.00
person_home_ownership                 3.00
loan_amnt                          7000.00
loan_intent                           3.00
loan_int_rate                        15.88
loan_percent_income                   0.08
cb_person_cred_hist_length            3.00
credit_score                        600.00
previous_loan_defaults_on_file        0.00
Name: 7225, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 1
--------------------------------------------------

'''

##now that we hve finished our parameter tunning and evaluated our model we can look into Random Forest
print("\n\nRandom Forest-----------------------------------------------------------------")
print("\nRandom Forest Data before numeric conversion-----")
print(RF_loan_data.head())
##For random Forest we will make sure all data is in numerical format
for column in categorical_columns:
    # Convert column to categorical type
    RF_loan_data[column] = RF_loan_data[column].astype("category")
    # Assign category codes to the column
    RF_loan_data[column] = RF_loan_data[column].cat.codes
    # Print unique codes for the column
    unique_values = RF_loan_data[column].unique()
    print(f"{column}: {unique_values}")
print("\nRandom Forest Data after numeric conversion-----")
'''
Unique values for categorical columns:
person_gender: [0 1] === ['female' 'male']
person_education: [4 3 1 0 2] === ['Master' 'High School' 'Bachelor' 'Associate' 'Doctorate']
                    0 = Associate
                    1 = Bachelor
                    2 = Doctorate
                    3 = High School
                    4 = Master
person_home_ownership: [3 2 0 1] === ['RENT' 'OWN' 'MORTGAGE' 'OTHER']
                    0 = MORTGAGE
                    1 = OTHER
                    2 = OWN
                    3 = RENT
loan_intent: [4 1 3 5 2 0] === ['PERSONAL' 'EDUCATION' 'MEDICAL' 'VENTURE' 'HOMEIMPROVEMENT' 'DEBTCONSOLIDATION']
                    0 = DEBTCONSOLIDATION
                    1 = EDUCATION
                    2 = HOMEIMPROVEMENT
                    3 = MEDICAL
                    4 = PERSONAL
                    5 = VENTURE
previous_loan_defaults_on_file: [0 1] === ['No' 'Yes']
                    0 = No
                    1 = Yes
'''
print(RF_loan_data.head())

##Train and test splitting
X = RF_loan_data.drop(columns=['loan_status'])
y = RF_loan_data['loan_status'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
# Train the model
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

y_pred = rf_model.predict(X_test)
# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.show()

from sklearn.model_selection import GridSearchCV
# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],       # Number of trees in the forest
    'max_depth': [10, 20, None],           # Maximum depth of trees
    'min_samples_split': [2, 5, 10],       # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum samples required at each leaf node
    'bootstrap': [True, False]             # Whether to use bootstrapping
}

# Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best Parameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# Use the best model with sample
best_rf_model = grid_search.best_estimator_
print("Test tuned model on test data-------------------------------------------------")
#Evaluate on the test set
y_test_pred = best_rf_model.predict(X_test)
print(f"Test Set Accuracy:{accuracy_score(y_test, y_test_pred)}")
print(f"Classification Report:\n {classification_report(y_test, y_test_pred)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


rejected_loans = RF_loan_data[LR_loan_data['loan_status'] == 0]
approved_loans = RF_loan_data[LR_loan_data['loan_status'] == 1]

# Randomly sample 5 from each group
sample_rejected = rejected_loans.sample(5, random_state=7)
sample_approved = approved_loans.sample(5, random_state=7)

# Combine the samples
sample_data = pd.concat([sample_rejected, sample_approved])


actual_vals = sample_data['loan_status']
sample_data = sample_data.drop(columns=['loan_status'])

predictions = best_rf_model.predict(sample_data)

# Display the results
print("\nPredictions vs Actual Values---------------------")
for i in range(len(sample_data)):
    print(f"Sample {i+1}:")
    print(f"Input Features:\n{sample_data.iloc[i]}")
    print(f"Predicted Loan Status (0 = Rejected, 1 = Approved): {predictions[i]}")
    print(f"Actual Loan Status: {actual_vals.iloc[i]}")
    print("-" * 50)

'''
Predictions vs Actual Values---------------------
Sample 1:
Input Features:
person_age                           26.00
person_gender                         1.00
person_education                      1.00
person_income                     98855.00
person_emp_exp                        5.00
person_home_ownership                 0.00
loan_amnt                          6000.00
loan_intent                           2.00
loan_int_rate                         7.90
loan_percent_income                   0.06
cb_person_cred_hist_length            4.00
credit_score                        702.00
previous_loan_defaults_on_file        0.00
Name: 12451, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 2:
Input Features:
person_age                           40.00
person_gender                         1.00
person_education                      1.00
person_income                     85896.00
person_emp_exp                       15.00
person_home_ownership                 0.00
loan_amnt                          7000.00
loan_intent                           1.00
loan_int_rate                        15.38
loan_percent_income                   0.08
cb_person_cred_hist_length           12.00
credit_score                        585.00
previous_loan_defaults_on_file        1.00
Name: 30959, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 3:
Input Features:
person_age                            35.00
person_gender                          0.00
person_education                       3.00
person_income                     144943.00
person_emp_exp                        11.00
person_home_ownership                  0.00
loan_amnt                           8000.00
loan_intent                            4.00
loan_int_rate                         11.01
loan_percent_income                    0.06
cb_person_cred_hist_length            10.00
credit_score                         591.00
previous_loan_defaults_on_file         1.00
Name: 26954, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 4:
Input Features:
person_age                            24.00
person_gender                          1.00
person_education                       4.00
person_income                     144985.00
person_emp_exp                         4.00
person_home_ownership                  0.00
loan_amnt                           6000.00
loan_intent                            1.00
loan_int_rate                          9.94
loan_percent_income                    0.04
cb_person_cred_hist_length             3.00
credit_score                         638.00
previous_loan_defaults_on_file         1.00
Name: 39742, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 5:
Input Features:
person_age                           54.00
person_gender                         0.00
person_education                      0.00
person_income                     35824.00
person_emp_exp                       31.00
person_home_ownership                 0.00
loan_amnt                          2500.00
loan_intent                           4.00
loan_int_rate                        11.71
loan_percent_income                   0.07
cb_person_cred_hist_length           28.00
credit_score                        626.00
previous_loan_defaults_on_file        1.00
Name: 32548, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 0
Actual Loan Status: 0
--------------------------------------------------
Sample 6:
Input Features:
person_age                           22.00
person_gender                         1.00
person_education                      3.00
person_income                     39160.00
person_emp_exp                        0.00
person_home_ownership                 3.00
loan_amnt                          6800.00
loan_intent                           5.00
loan_int_rate                        14.11
loan_percent_income                   0.17
cb_person_cred_hist_length            2.00
credit_score                        559.00
previous_loan_defaults_on_file        0.00
Name: 7475, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 1
Actual Loan Status: 1
--------------------------------------------------
Sample 7:
Input Features:
person_age                           26.00
person_gender                         1.00
person_education                      3.00
person_income                     57199.00
person_emp_exp                        2.00
person_home_ownership                 3.00
loan_amnt                         15000.00
loan_intent                           4.00
loan_int_rate                         7.90
loan_percent_income                   0.26
cb_person_cred_hist_length            3.00
credit_score                        586.00
previous_loan_defaults_on_file        0.00
Name: 17093, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 1
Actual Loan Status: 1
--------------------------------------------------
Sample 8:
Input Features:
person_age                            22.00
person_gender                          1.00
person_education                       4.00
person_income                     130852.00
person_emp_exp                         0.00
person_home_ownership                  3.00
loan_amnt                          28000.00
loan_intent                            1.00
loan_int_rate                         11.01
loan_percent_income                    0.21
cb_person_cred_hist_length             4.00
credit_score                         683.00
previous_loan_defaults_on_file         0.00
Name: 85, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 1
Actual Loan Status: 1
--------------------------------------------------
Sample 9:
Input Features:
person_age                           24.00
person_gender                         1.00
person_education                      3.00
person_income                     29794.00
person_emp_exp                        5.00
person_home_ownership                 3.00
loan_amnt                         10000.00
loan_intent                           5.00
loan_int_rate                         8.70
loan_percent_income                   0.34
cb_person_cred_hist_length            2.00
credit_score                        534.00
previous_loan_defaults_on_file        0.00
Name: 10372, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 1
Actual Loan Status: 1
--------------------------------------------------
Sample 10:
Input Features:
person_age                           23.00
person_gender                         1.00
person_education                      0.00
person_income                     53888.00
person_emp_exp                        4.00
person_home_ownership                 3.00
loan_amnt                          8000.00
loan_intent                           1.00
loan_int_rate                        13.49
loan_percent_income                   0.15
cb_person_cred_hist_length            2.00
credit_score                        637.00
previous_loan_defaults_on_file        0.00
Name: 8945, dtype: float64
Predicted Loan Status (0 = Rejected, 1 = Approved): 1
Actual Loan Status: 1
--------------------------------------------------

'''