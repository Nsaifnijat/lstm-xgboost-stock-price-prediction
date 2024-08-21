import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import xgboost as xgb

# Load data
df = pd.read_csv('uncorrelatedfeatures.csv', index_col='time', parse_dates=True)
df = df.dropna()

# Create target variable
df['tomorrow'] = df['close'].shift(-7)
df['target'] = (df['tomorrow'] > df['close']).astype(int)

# Handle missing and infinite values
df.replace([np.nan, np.inf, -np.inf], -99999, inplace=True)

# Drop columns not used for training
df_train = df.drop(columns=['tomorrow', 'target'])
target = df['target']

# Split the data
train_size = int(len(df) * 0.8)
X_train = df_train[:train_size]
X_test = df_train[train_size:]
y_train = target[:train_size]
y_test = target[train_size:]

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training using XGBoost
model = xgb.XGBClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
preds = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
conf_matrix = confusion_matrix(y_test, preds)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print('Confusion Matrix:')
print(conf_matrix)
