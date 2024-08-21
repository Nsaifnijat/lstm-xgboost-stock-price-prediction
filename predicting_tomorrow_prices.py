import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('uncorrelatedfeatures.csv', index_col='time', parse_dates=True)

df = df.dropna()

df['tomorrow'] = df['close'].shift(-7)
df['target'] = (df['tomorrow']  > df['close']).astype(int)


from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train_size = int(len(df)*0.8)


df.replace([np.nan,np.inf, -np.inf], -99999, inplace=True)
df_train = df.drop(columns=['tomorrow','target'])
target = df['target']

#scaling in a way that sum of each row becomes one
df_normalized = preprocessing.normalize(df_train,norm='l1')
df_normalized = pd.DataFrame(df_normalized, columns=df_train.columns)
df_normalized.index = df_train.index


X_train = df_normalized[:train_size]
X_test = df_normalized[train_size:]

y_train = df['target'][:train_size]
y_test = df['target'][train_size:]


model.fit(X_train, y_train)

from sklearn.metrics import precision_score

preds = model.predict(X_test)
preds = pd.Series(preds, index=y_test.index)
accuracy = precision_score(y_test, preds)

print(accuracy)