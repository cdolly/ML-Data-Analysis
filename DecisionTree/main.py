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


df = pandas.read_csv("shows.csv")

print(df)
