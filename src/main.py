import matplotlib.pyplot as plt
import pandas as pd
import numpy 
import scipy
#Necessary imports for this project


#Generate 2 arrays of normally distributed data, each with 100 entries
#Though randomly created, the two variables are meant to represent customers shopping habits
#X represents the minutes spent beforemaking a purchase
#Y represents the cost of the purchase made
x = numpy.random.normal(3, 1, 100) 
y = numpy.random.normal(150, 40, 100) / x


#Splitting the data into the testing training sets

trainX = x[:75] # 75% of the data goes into the training set
trainY = y[:75]

testX = x[75:] # The remaining 25% goes into the testing set
testY = y[75:]

# Visualization of the training sets through a scatter plot.
plt.scatter(trainX, trainY)
plt.show()
