import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('liver_dataset.csv')

# Preprocess (example)
data = data.dropna()
X = data.drop('LiverCirrhosis', axis=1)  # Replace with actual target column
y = data['LiverCirrhosis']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('model/liver_model.pkl', 'wb') as f:
    pickle.dump(model, f)