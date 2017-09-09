"""
=====
SMOTE
=====

An illustration of the SMOTE method and its variant.

"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import csv
from imblearn.over_sampling import SMOTE

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
#----------------------------------------------------------------------------------------------------
#import the data
#----------------------------------------------------------------------------------------------------

df = pd.read_csv("data with day.csv", encoding = "ISO-8859-1")
data_clean = df.dropna()
X = data_clean[[   'nWarnings', 'nAlarm',  'Friday',   'Monday',   'Saturday', 'Sunday',   'Thursday', 'Tuesday',  'Wednesday']]
y = data_clean.departure
print('Original dataset shape {}'.format(Counter(y)))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X, y)

np.savetxt('X_res.csv', X_res, delimiter=',')
np.savetxt('y_res.csv', y_res, delimiter=',')

print('Resampled dataset shape {}'.format(Counter(y_res)))

# Apply regular SMOTE
# kind = ['regular', 'borderline1', 'borderline2', 'svm']
# sm = [SMOTE(kind=k) for k in kind]
# X_resampled = []
# y_resampled = []
# X_res_vis = []
# for method in sm:
#     X_res, y_res = method.fit_sample(X, y)
#     X_resampled.append(X_res)
#     y_resampled.append(y_res)
# #X_resampled.append(y_resampled)
#     #X_res_vis.append(pca.transform(X_res))
# with open("X_res.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(X_res)

# with open("y_res.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(y_resampled)

#print('Resampled dataset shape {}'.format(Counter(y_resampled)))























