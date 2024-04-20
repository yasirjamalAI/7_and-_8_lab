from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Introduce some random noise into the dataset
import numpy as np
np.random.seed(42)
X_noise = X + np.random.normal(0, 0.5, size=X.shape)

# Select only the first two features to reduce complexity
X_reduced = X_noise[:, :2]

# Split the noisy dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Initialize a logistic regression classifier
log_reg = LogisticRegression(max_iter=1000)

# Train the classifier on the training set
log_reg.fit(X_train, y_train)

# Predict labels for the test set
y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print("Accuracy:", accuracy,"100")
