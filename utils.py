import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

def preprocess(df):
    # **ADD YOUR CODE HERE**
    
    #split outcome column from rest of dataset to preserve it
    features = df.drop(columns='Outcome')
    target = df['Outcome']

    #imputation replacing 0 vals with column mean
    imp_mean = SimpleImputer(missing_values=0, strategy='mean')
    features = pd.DataFrame(imp_mean.fit_transform(features), columns=features.columns)

    #print out remaing number of 0s to test imputation
    for col in features.columns:
        print("Number of 0 values in column "+col+":")
        print((features[col] == 0).sum())

    #standardize the dataset
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)

    #add back outcome column
    df['Outcome'] = target

    return df