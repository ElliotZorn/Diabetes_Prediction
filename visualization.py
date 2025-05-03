import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
#import utils

try:
    df = pd.read_csv('diabetes.csv')
except:
    print("""
      Dataset not found in your computer.
      Please follow the instructions on how to download the dataset.
      """)
    quit()

# **ADD YOUR CODE HERE**

print("First 5 rows of the dataset:")
print(df.head())

print("\nStatistical summary of dataset:")
print(df.describe())

# assuming missing means 0 and not null.
print("\nNumber of zero values for each variable:")
print((df == 0).sum())

#test utils preprocess
#df = utils.preprocess(df)

# keep this at end, as it blocks execution
print("\nDisplaying histograms for each variable:")
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()
