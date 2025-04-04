import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
DATA_PATH = os.path.join('E:', 'data.pickle')

with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)

# Convert data to a uniform format
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Ensure all sequences have the same length
max_length = max(len(seq) for seq in data)  # Find the longest sequence
data_padded = [seq + [0] * (max_length - len(seq)) for seq in data]  # Pad shorter sequences

# Convert to NumPy array
data_array = np.asarray(data_padded, dtype=np.float32)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

print(f'✅ Accuracy: {accuracy * 100:.2f}% of samples classified correctly!')

# Save trained model
MODEL_PATH = 'model.p'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f'✅ Model saved successfully as {MODEL_PATH}!')
a
