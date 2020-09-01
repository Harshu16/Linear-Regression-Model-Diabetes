import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use all  features
diabetes_X= diabetes.data

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# Split the targets into training/testing sets
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]
# Create linear regression object
model= linear_model.LinearRegression()
# Train the model using the training sets
model.fit(diabetes_X_train,diabetes_Y_train)
# Make predictions using the testing set
diabetes_Y_pred = model.predict(diabetes_X_test)
# In y=mx+b.....weight=m,feature=x,intercept=b.
print("Coefficient of linear regression or weights is:",model.coef_)
print("intercepts is:",model.intercept_)
# The mean squared error=The mean squared error tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. The squaring is necessary to remove any negative signs.
print('Mean Squared error: %.2f'%mean_squared_error(diabetes_Y_test,diabetes_Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of prediction: %.2f'%r2_score(diabetes_Y_test,diabetes_Y_pred))
# Plot outputs
#plt.scatter(diabetes_X_test,diabetes_Y_test)
#plt.plot(diabetes_X_test,diabetes_Y_pred)
#plt.show()
#Output by extracting just one feature for plotting staight line y=mx+b is
#Coefficient of linear regression or weights is: [941.43097333]
#intercepts is: 153.39713623331698
#Mean Squared error: 3035.06
#Coefficient of prediction: 0.41
#Output by extracting all features is:
#Coefficient of linear regression or weights is: [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
  458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
#intercepts is: 153.05827988224112
#Mean Squared error: 1826.54
#Coefficient of prediction: 0.65
