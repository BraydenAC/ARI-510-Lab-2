import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import pandas as pd
import numpy as np

#functions
def StringToFloats(input, output=None):
    if output is None:
        output = []

    #iterate through array of strings
    for string in input:
        # Turn the string into an array of single element strings
        string = string.lstrip()
        print(string)
        splitArray = string.split(' ')

        # Convert string elements to float elements
        floatArray = [float(element) for element in splitArray]

        # Append the float array to the output list
        output.append(floatArray)

    return output


#Loading dataset into project
X_train_dev = np.array(StringToFloats(pd.read_csv('UCI HAR Dataset/train/X_train.txt')))
y_train_dev = np.array(StringToFloats(pd.read_csv('UCI HAR Dataset/train/y_train.txt')))
X_test =      np.array(StringToFloats(pd.read_csv('UCI HAR Dataset/test/X_test.txt')))
y_test =      np.array(StringToFloats(pd.read_csv('UCI HAR Dataset/test/y_test.txt')))

#turn string array's into matrix and vector counterparts


#Split train into train and dev
X_train, X_dev, y_train, y_dev= train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=42)

#Apply standardization to the data
scaled_X_train = scaler.fit_transform(X_train)
scaled_y_train = scaler.fit_transform(y_train)
scaled_X_dev = scaler.transform(X_dev)
scaled_y_dev = scaler.transform(y_dev)
scaled_X_test = scaler.transform(X_test)
scaled_y_test = scaler.transform(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train)
