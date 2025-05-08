import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(16)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from utils import preprocess

# Load dataset
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    print("Dataset not found. Please follow the instructions to download the dataset.")
    quit()


# Perform preprocessing (imputation, standardization)
df = preprocess(df)

# Split the data into a training and testing set
X=df.drop(columns=['Outcome'])
y=df['Outcome']

# Train/test split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("training set size:", x_train.shape)

# Build MLP
model = Sequential(
        [Dense(32,activation='relu',name='layer1',input_shape=(X.shape[1],)),
         Dense(16,activation='relu',name='layer2'),
         Dropout(0.3),
         Dense(8,activation='relu',name='layer3'),
         Dropout(0.2),
         Dense(1,activation='sigmoid',name='output'),]
        )
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
model.fit(
    x_train,y_train,
    epochs=100,
    batch_size=16,
    validation_data=(x_test,y_test),
    callbacks=[early_stop],
    verbose=1
    )
model.evaluate(x_test,y_test)
# Results - Accuracy
