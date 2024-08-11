# Ybi-foundation-project-report-"Bank Customer Churn Prediction"
Title of Project:- 
"Bank Customer Churn Prediction"

Objective:- 

The objective of this project is to develop a predictive model to identify bank customers who are likely to churn. This will enable the bank to implement targeted retention strategies and improve customer loyalty.

Data Source:- 

The dataset used in this project is the "Bank Customer Churn Dataset" available on Kaggle. Kaggle Link.

Import Library:- 

python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
Import Data
python
df = pd.read_csv('Churn_Modelling.csv')
df.head()
Describe Data
python
df.info()
df.describe()

Data Visualization:- 

sns.countplot(x='Exited', data=df)
plt.title('Distribution of Churn')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
Data Preprocessing:- 

# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)

# Standardize the features
scaler = StandardScaler()
df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']])
df.head()

Define Target Variable (y) and Feature Variables (X):-

# Define the target variable and feature variables
X = df.drop('Exited', axis=1)
y = df['Exited']

X.head(), y.head()

Train Test Split:- 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

Modeling:- 

# Initialize the models
log_reg = LogisticRegression()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Train the models
log_reg.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Get predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_decision_tree = decision_tree.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)

Model Evaluation :- 

# Evaluate the Logistic Regression model
print("Logistic Regression Model Evaluation:")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Evaluate the Decision Tree model
print("Decision Tree Model Evaluation:")
print(confusion_matrix(y_test, y_pred_decision_tree))
print(classification_report(y_test, y_pred_decision_tree))

# Evaluate the Random Forest model
print("Random Forest Model Evaluation:")
print(confusion_matrix(y_test, y_pred_random_forest))
print(classification_report(y_test, y_pred_random_forest))

Prediction:- 


# Make predictions with the Random Forest model
rf_predictions = random_forest.predict(X_test)

# Display some predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': rf_predictions})
results.head()

Explaination :- 
Model Explanation:

Logistic Regression: It is a linear model used for binary classification. It estimates the probability of the default class. The coefficients of the model indicate the influence of each feature on the probability of a customer churning.
Decision Tree Classifier: This model splits the data into branches to make decisions based on the features. It is easy to interpret but may overfit the data.
Random Forest Classifier: An ensemble method that uses multiple decision trees to improve classification accuracy and control overfitting.
