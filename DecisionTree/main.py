# Machine Learning with Python
# Created by Cameron Dolly
# Goal is to create a simulation of what using a decision tree to formulate a chouce using an example data set.
# The idea for this project was obtained through the internet, though the coding and work are my own

import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
# Necessary imports for this project


# The dataset for this project is a set of comedians with their age, experience, rank, and nationality as well as whether you should go to the show or not.
#   We will use a decision tree to decide whether you should go to a given comedians show, given their age, experience, rank and nationality.
df = pd.read_csv("shows.csv") # Creates a dataframe from the CSV file

# All data must be numerical for a decision tree, so we must convert the categorical data into numerical data.
s = {'UK': 0, 'USA': 1, 'N': 2} # Creates a map that "translates" the Strings UK, USA, ad N, to 0, 1, and 2 respectively
df['Nationality'] = df['Nationality'].map(s) # Applies the map to the DataFrame
s = {'YES': 1, 'NO': 0} # Creates a map similarly to before, translating YES and NO to 1 and 0 respectively.
df['Go'] = df['Go'].map(s) # Applies the map to the DataFrame


# Next, we must seperate the columns we wish to use to predict from, and the ones we wish to predict.
features = ['Age', 'Experience', 'Rank', 'Nationality'] # features is used to denote the columns seperated for use of prediction

x = df[features] # The DataFrame containing features
y = df['Go'] # The DataFrame containing what we wish to predict, which is only 1 column, whether we will go to the comedy show or not.

 # print(x)
 # print("------------------------------------------------") # Visualizes the two new DataFrames and their respective columns.
 # print(y)

dTree = DecisionTreeClassifier() # Creates a decision tree classifier object
dTree = dTree.fit(x, y) # Fits the two variables to the decision tree
data = tree.export_graphviz(dtree, out_file=None, feature_names=features) # Creates the visualization of the decision tree
graph = pydotplus.graph_from_dot_data(data) 
graph.write_png('mydecisiontree.png') # Saves the png to your computer

img=pltimg.imread('mydecisiontree.png') # reads the png from your computer and saves it to a variable
imgplot = plt.imshow(img) # Plots the image
plt.show()

# Now onto the results
# The first number represents what the threshold for that given value is, if the number is smaller than the given value, then they will follow True,
#   and otherwise false.
# gini refers to how well the features were split. It is always a number between 0.0 and 0.5, with 0.0 meaning all results were the same and 0.5 meaning 50% of
#   results were true, and the other 50% false.
# Samples represents how many entries were analyzed at this given point in the decision tree.
# Value, represented as Value = [x, y], illustrates how many of the samples were True or False, with x representing False and y representing True.

# Now we can use the decision tree to predict any new values
# This will print either a 1 or a 0, meaing Go or Do not go depending on the input.
print(dtree.predict([[40, 10, 7, 1]]))
