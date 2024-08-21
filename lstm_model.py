import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data
df = pd.read_csv('uncorrelatedfeatures.csv', index_col='time', parse_dates=True)
df = df.dropna()

# Create target variable
df['tomorrow'] = df['close'].shift(-7)
df['target'] = (df['tomorrow'] > df['close']).astype(int)
df.dropna(inplace=True)  # Drop rows with NaN values created by the shift

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

# Create sequences for LSTM
lookback = 30  # Lookback period
batch_size = 32

train_gen = TimeseriesGenerator(X_train_scaled, y_train, length=lookback, batch_size=batch_size)
test_gen = TimeseriesGenerator(X_test_scaled, y_test, length=lookback, batch_size=batch_size)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_gen, epochs=50, validation_data=test_gen)

# Evaluate the model
preds = (model.predict(test_gen) > 0.5).astype("int32").flatten()
accuracy = accuracy_score(y_test[lookback:], preds)
precision = precision_score(y_test[lookback:], preds)
conf_matrix = confusion_matrix(y_test[lookback:], preds)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print('Confusion Matrix:')
print(conf_matrix)
