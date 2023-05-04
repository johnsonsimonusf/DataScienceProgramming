import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
# import os
# exit(os.getcwd())

mower_model = pickle.load(open('./data/risingmower_svm_model.pkl', "rb"))

print("\n*************************************************************")
print("*** The USF Super Simple Lawn Mower Ownership Prediction Model *")
print("***************************************************************\n")
income = int(input("Enter income: "))
lot_size = float(input("Enter lot size: "))
df = pd.DataFrame({'Income': [income]}, {'Lot_Size': [lot_size]})
result = mower_model.predict(df)
probability = mower_model.predict_proba(df)
ownership = ('Non owner', 'Owner')
print(f"\nThe USF Simple Lawn Mower Prediction model indicates probability of Ownership at {probability[0][1]:.4f}, therefore it's indicated that we should {ownership[result[0]]}.\n")
