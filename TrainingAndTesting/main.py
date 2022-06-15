# Machine Learning with Python
# Created by Cameron Dolly
# Goal is to create a simulation of what training and testing a dataset would look like using a randomly generated, normally distributed, data set.
# The idea for this project was obtained through the internet, though the coding and work are my own

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.metrics import r2_score
#Necessary imports for this project


# Generate 2 arrays of normally distributed data, each with 100 entries
# Though randomly created, the two variables are meant to represent customers shopping habits
# X represents the minutes spent beforemaking a purchase
# Y represents the cost of the purchase made
x = np.random.normal(3, 1, 100) # Mean of 3, std of 1, 100 entries
y = np.random.normal(150, 40, 100) / x # mean of 150, std of 40, 100 entries


# Splitting the data into the testing training sets

trainX = x[:75] # 75% of the data goes into the training set
trainY = y[:75]

testX = x[75:] # The remaining 25% goes into the testing set
testY = y[75:]

# Visualization of the training sets through a scatter plot.
# Represented in blue dots
 # plt.scatter(trainX, trainY)
 # plt.show()

# Visualization of the testing set, should vaguely represent the original scatter plot as well as the training set.
# Represented in orange dots
 # plt.scatter(testX, testY)
 # plt.show()

# Though a linear regression could be used, the distribution appears to be more polynomial-like.
polyRegressionModel = np.poly1d(np.polyfit(trainX, trainY, 4)) # Creates the polynomial model using the training set x and y values.
polyRegressionLine = np.linspace(0, 6, 100) # Creates the line using the polynomial model.

# Visualization of the training set and the polynomial regression line.
plt.scatter(trainX, trainY)
plt.plot(polyRegressionLine, polyRegressionModel(polyRegressionLine))
plt.show()
 
# r2, meaning r squared, gives us an indicator of how well the data fits the polynominal model
# The higher this value is the better the data fits the model. 
# The result will be different everytime this script is run due to the randomly generated dataset.
r2 = r2_score(trainY, polyRegressionModel(trainX))
print('The r^2 value of the training set is ...')
print(r2)

# Now to see how well the testing set compares to the model that we have created.
# If the r squared value is negative, that means that the model created actually fits the testing set worse than a horizontal line does.
# This is due to the lack of a constant term in the polynomial equation.
testr2 = r2_score(testY, polyRegressionModel(testX))
print('The r^2 value of the testing set is ...')
print(testr2)

# If the model created is a good fit for the test set, we can use it to predict future values
# This prints what amount of money the customer would be likely to spend if they spent 3 minutes shopping.
print(polyRegressionModel(3))

# As stated prior, due to the random generation of the dataset, the results will vary significantly, but the closer the r^2 value
#   of the testing set is to 1.0, the better.
