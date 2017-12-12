import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the Autemobile dataset
Autemobile = pd.read_csv('datasets.csv')

# Use only one feature
# Autemobile_X = Autemobile.data[:, np.newaxis, 2]

Autemobile_X = Autemobile.drop("price",axis=1)

# Split the data into training/testing sets
Autemobile_X_train = Autemobile_X[:-20]
Autemobile_X_test = Autemobile_X[-20:]

# Split the targets into training/testing sets
Autemobile_y_train = Autemobile.price[:-20]
Autemobile_y_test = Autemobile.price[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(Autemobile_X_train, Autemobile_y_train)

# Make predictions using the testing set
Autemobile_y_pred = regr.predict(Autemobile_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Autemobile_y_test, Autemobile_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Autemobile_y_test, Autemobile_y_pred))

print(Autemobile_y_pred)
print(Autemobile_y_test)
print(Autemobile_y_test - Autemobile_y_pred)

# Plot outputs
# plt.scatter(Autemobile_X_test, Autemobile_y_test,  color='black')
# plt.plot(Autemobile_X_test, Autemobile_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
