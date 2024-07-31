# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading the data")

data = pd.read_csv('/Users/tanaysaxena/Documents/Coding/ADG_ML/Task_2/Fraud.csv')

print("\nChecking for missing data...")
print(data.isnull().sum())
# Get some basic statis on the data
print("\nLet's take a look at some summary statistics...")
print(data.describe())
# How maany fraudulent transactions are there? Let's visualize it!
print("\nDistribution of fraudulent transactions...")
sns.countplot(x='isFraud', data=data)  # Updated column name
plt.title('Distribution of Fraudulent Transactions')
plt.show()
print("\nRelationships between features...")
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
print("\nEncoding categorical data...")
data = pd.get_dummies(data, columns=['type'], drop_first=True)  # Encoding categorical 'type' column
# Split the data into features (X) and target variable (y)
X = data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)  # Exclude non-numeric columns if necessary
y = data['isFraud']
print("\nSplitting data into training and testing setss")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nAddressing imbalanced class distribution...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print("\nTraining a Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"\nROC-AUC Score: {roc_auc}")
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)
print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
best_model.fit(X_train_res, y_train_res)
y_pred_best = best_model.predict(X_test_scaled)
print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred_best))
print("\nBest Model Confusion Matrix:")
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues')
plt.title('Best Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
roc_auc_best = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
print(f"\nBest Model ROC-AUC Score: {roc_auc_best}")
fpr_best, tpr_best, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr_best, tpr_best, label=f'Best ROC curve (area = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best ROC Curve')
plt.legend(loc='best')
plt.show()
feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

#This gives me the desired output from the required dataset
#if sir you want to run this file then please change the path accordingly to your system and the place where you downloaded the dataset.

