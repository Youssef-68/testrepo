import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#to avoid warning
import warninigs
warningsfilterwarnings('ignore')

house = pd.read_csv("/kaggle/input/the-boston-housing-dataset/Boston (1).csv")
house.head()

#data information
house.info

#check of duplicated values
house.duplicated().sum()

#summary of data
house.describe
