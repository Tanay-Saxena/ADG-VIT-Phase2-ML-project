import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Let's load our insurance data, stored in a CSV file
data = pd.read_csv('/Users/tanaysaxena/Documents/Coding/ADG_ML/Task_1/insurance.csv')

# Peek at the first few rows to get a sense of the data
print("First glimpse of our data:")
print(data.head())

# Check if any data points are missing in action (missing values)
print("\nAre there any missing values? Let's see:")
print(data.isnull().sum())

# Time to visualize the spread of 'charges' (our target variable)
plt.figure(figsize=(8,5))
sns.histplot(data['charges'],kde=True)  # Plot the distribution with a smooth density curve
plt.title('Distribution of Charges (How often do costs occur?)')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# Let's see how the variables are connected - correlation matrix to the rescue!
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')  # Show correlation values and use a coolwarm colormap
plt.title('Correlation Matrix (How do variables influence each other?)')
plt.show()

# We need to convert non-numerical variables like 'sex' or 'region' into numbers the computer understands
# One-hot encoding does the trick!
data_encoded = pd.get_dummies(data,columns=['sex', 'smoker', 'region'],drop_first=True)

# Separate our features (X) from the target variable (y) - 'charges' is what we're predicting
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

# Split the data into training and testing sets (80% for training, 20% for testing)
# This helps us assess how well our model generalizes to unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to have a mean of 0 and a variance of 1 (helps the model learn better)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Let's train a Linear Regression model! This model learns the linear relationships between features and the target variable
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Now that the model is trained, let's use it to predict charges on the test data
y_pred = model.predict(X_test_scaled)

# How well did we do? Let's calculate the Root Mean Squared Error (RMSE)
# This tells us the average difference between the actual charges and our predictions (lower is better)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'How well did we predict charges? Root Mean Squared Error: {rmse}')

# Another metric: R-squared. It tells us how much of the variance in the target variable is explained by the model (higher is better)
r2 = r2_score(y_test, y_pred)
print(f'How well does the model explain the charges? R-squared: {r2}')

# Let's see how much each feature contributes to the model's predictions - coefficients come in handy!
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nCoefficients of the linear regression model (how much each feature influences charges):")
print(coefficients)

# Permutation Importance helps us understand how important each feature is for the model's predictions
# We shuffle a feature's values and see how much the model's performance drops - the more it drops, the more important the feature
perm_importance = permutation_importance(model, X_test)

#This gives me the desired output from the required dataset
#if sir you want to run this file then please change the path accordingly to your system and the place where you downloaded the dataset.s