# This is the train file

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Read CSV data (IRIS dataset)
data_path = "data/iris.csv"  # Replace with your dataset path
df = pd.read_csv(data_path)

# Assuming the dataset has the standard Iris columns
X = df.drop('species', axis=1)
y = df['species']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build a model pipeline with a Decision Tree
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# 3. Evaluate and log metrics
y_pred = pipeline.predict(X_test)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_macro': precision_score(y_test, y_pred, average='macro'),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
    'f1_macro': f1_score(y_test, y_pred, average='macro')
}

# Save metrics to CSV
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('metrics.csv', index=False)

# 4. Save the model
joblib.dump(pipeline, 'model.h5')

print("Model training complete. Metrics and model saved successfully.")